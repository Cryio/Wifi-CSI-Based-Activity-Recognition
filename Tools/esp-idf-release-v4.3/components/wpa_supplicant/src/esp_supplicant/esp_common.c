/*
 * SPDX-FileCopyrightText: 2020-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils/includes.h"
#include "utils/common.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_wifi_driver.h"
#include "drivers/driver.h"
#include "common/bss.h"
#include "common/rrm.h"
#include "common/wnm_sta.h"
#include "common/wpa_supplicant_i.h"
#include "esp_supplicant/esp_scan_i.h"
#include "esp_supplicant/esp_common_i.h"
#include "common/ieee802_11_common.h"
#include "esp_rrm.h"
#include "esp_wnm.h"
#include "rsn_supp/wpa.h"
#include "esp_private/wifi.h"


struct wpa_supplicant g_wpa_supp;

#if defined(CONFIG_WPA_11KV_SUPPORT)
static TaskHandle_t s_supplicant_task_hdl = NULL;
static void *s_supplicant_evt_queue = NULL;
static void *s_supplicant_api_lock = NULL;
static bool s_supplicant_task_init_done;

static int handle_action_frm(u8 *frame, size_t len,
			     u8 *sender, int8_t rssi, u8 channel)
{
	struct ieee_mgmt_frame *frm = os_malloc(sizeof(struct ieee_mgmt_frame) + len);

	if (!frm) {
		wpa_printf(MSG_ERROR, "memory allocation failed");
		return -1;
	}

	os_memcpy(frm->sender, sender, ETH_ALEN);
	frm->len = len;
	frm->channel = channel;
	frm->rssi = rssi;

	os_memcpy(frm->payload, frame, len);
	if (esp_supplicant_post_evt(SIG_SUPPLICANT_RX_ACTION, (u32)frm) != 0) {
		os_free(frm);
		return -1;
	}

	return 0;
}

static void handle_rrm_frame(struct wpa_supplicant *wpa_s, u8 *sender,
			     u8 *payload, size_t len, int8_t rssi)
{
	if (payload[0] == WLAN_RRM_NEIGHBOR_REPORT_RESPONSE) {
		/* neighbor report parsing */
		wpas_rrm_process_neighbor_rep(wpa_s, payload + 1, len - 1);
	} else if (payload[0] == WLAN_RRM_RADIO_MEASUREMENT_REQUEST) {
		/* Beacon measurement */
		wpas_rrm_handle_radio_measurement_request(wpa_s, NULL,
				sender, payload + 1, len - 1);
	} else if (payload[0] == WLAN_RRM_LINK_MEASUREMENT_REQUEST) {
		/* Link measurement */
		wpas_rrm_handle_link_measurement_request(wpa_s, NULL,
				payload + 1, len - 1, rssi);
	}
}

static int mgmt_rx_action(u8 *sender, u8 *payload, size_t len, u8 channel, int8_t rssi)
{
	u8 category;
	u8 bssid[ETH_ALEN];
	struct wpa_supplicant *wpa_s = &g_wpa_supp;
	int ret = esp_wifi_get_assoc_bssid_internal(bssid);

	if (ret < 0) {
		wpa_printf(MSG_INFO, "STA not associated");
		return -1;
	}

	category = *payload++;
	len--;
	if (category == WLAN_ACTION_WNM) {
		ieee802_11_rx_wnm_action(wpa_s, sender, payload, len);
	} else if (category == WLAN_ACTION_RADIO_MEASUREMENT) {
		handle_rrm_frame(wpa_s, sender, payload, len, rssi);
	}

	return 0;
}

static void btm_rrm_task(void *pvParameters)
{
	supplicant_event_t *evt;
	bool task_del = false;

	while(1) {
		if (xQueueReceive(s_supplicant_evt_queue, &evt, portMAX_DELAY) != pdTRUE)
			continue;

		/* event validation failed */
		if (evt->id >= SIG_SUPPLICANT_MAX) {
			os_free(evt);
			continue;
		}

		switch (evt->id) {
		case SIG_SUPPLICANT_RX_ACTION:
		{
			struct ieee_mgmt_frame *frm = (struct ieee_mgmt_frame *)evt->data;
			mgmt_rx_action(frm->sender, frm->payload, frm->len, frm->channel, frm->rssi);
			os_free(frm);
			break;
		}

		case SIG_SUPPLICANT_SCAN_DONE:
			esp_supplicant_handle_scan_done_evt();
			break;
		case SIG_SUPPLICANT_DEL_TASK:
			task_del = true;
			break;
		default:
			break;
		}

		os_free(evt);

		if (task_del)
			break;
	}

	vQueueDelete(s_supplicant_evt_queue);
	s_supplicant_evt_queue = NULL;

	/* At this point, we completed */
	vTaskDelete(NULL);
}

static void clear_bssid_flag(struct wpa_supplicant *wpa_s)
{
	wifi_config_t *config;

	/* Reset only if btm is enabled */
	if (esp_wifi_is_btm_enabled_internal(WIFI_IF_STA) == false)
		return;

	config = os_zalloc(sizeof(wifi_config_t));

	if (!config) {
		wpa_printf(MSG_ERROR, "failed to allocate memory");
		return;
	}

	esp_wifi_get_config(WIFI_IF_STA, config);
	if (config->sta.bssid_set) {
		config->sta.bssid_set = 0;
		esp_wifi_set_config(WIFI_IF_STA, config);
	}
	os_free(config);
	wpa_printf(MSG_DEBUG, "cleared bssid flag");
}

static void register_action_frame(struct wpa_supplicant *wpa_s)
{
	wpa_s->type &= ~(1 << WLAN_FC_STYPE_ACTION);
	/* subtype is defined only for action frame */
	wpa_s->subtype = 0;

	/* current supported features in supplicant: rrm and btm */
	if (esp_wifi_is_rm_enabled_internal(WIFI_IF_STA))
		wpa_s->subtype = 1 << WLAN_ACTION_RADIO_MEASUREMENT;
	if (esp_wifi_is_btm_enabled_internal(WIFI_IF_STA))
		wpa_s->subtype |= 1 << WLAN_ACTION_WNM;

	if (wpa_s->subtype)
		wpa_s->type |= 1 << WLAN_FC_STYPE_ACTION;

	esp_wifi_register_mgmt_frame_internal(wpa_s->type, wpa_s->subtype);
}

#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */

void esp_supplicant_unset_all_appie(void)
{
   uint8_t appie;
   for (appie = WIFI_APPIE_PROBEREQ; appie < WIFI_APPIE_RAM_MAX; appie++) {
        esp_wifi_unset_appie_internal(appie);
   }
}

static int ieee80211_handle_rx_frm(u8 type, u8 *frame, size_t len, u8 *sender,
				   int8_t rssi, u8 channel, u64 current_tsf)
{
	int ret = 0;

	switch (type) {
#if defined(CONFIG_WPA_11KV_SUPPORT)
	case WLAN_FC_STYPE_BEACON:
	case WLAN_FC_STYPE_PROBE_RESP:
		ret = esp_handle_beacon_probe(type, frame, len, sender, rssi, channel, current_tsf);
		break;
#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */
	case WLAN_FC_STYPE_ASSOC_RESP:
	case WLAN_FC_STYPE_REASSOC_RESP:
		wpa_sm_notify_assoc(&gWpaSm, sender);
		break;
#if defined(CONFIG_WPA_11KV_SUPPORT)
	case WLAN_FC_STYPE_ACTION:
		ret = handle_action_frm(frame, len, sender, rssi, channel);
		break;
#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */
	default:
		ret = -1;
		break;
	}

	return ret;
}

int esp_supplicant_common_init(struct wpa_funcs *wpa_cb)
{
	struct wpa_supplicant *wpa_s = &g_wpa_supp;
	int ret;

#if defined(CONFIG_WPA_11KV_SUPPORT)
	if (!s_supplicant_api_lock) {
		s_supplicant_api_lock = xSemaphoreCreateRecursiveMutex();
	}
	if (!s_supplicant_api_lock) {
		wpa_printf(MSG_ERROR, "%s: failed to create Supplicant API lock", __func__);
		ret = -1;
		goto err;
	}

	s_supplicant_evt_queue = xQueueCreate(3, sizeof(supplicant_event_t));

	if (!s_supplicant_evt_queue) {
		wpa_printf(MSG_ERROR, "%s: failed to create Supplicant event queue", __func__);
		ret = -1;
		goto err;
	}
	ret = xTaskCreate(btm_rrm_task, "btm_rrm_t", SUPPLICANT_TASK_STACK_SIZE, NULL, 2, &s_supplicant_task_hdl);
	if (ret != pdPASS) {
		wpa_printf(MSG_ERROR, "btm: failed to create task");
		ret = -1;
		goto err;
	}

	s_supplicant_task_init_done = true;
	esp_scan_init(wpa_s);
	wpas_rrm_reset(wpa_s);
	wpas_clear_beacon_rep_data(wpa_s);

#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */
	wpa_s->type = 0;
	wpa_s->subtype = 0;
	wpa_s->type |= (1 << WLAN_FC_STYPE_ASSOC_RESP) | (1 << WLAN_FC_STYPE_REASSOC_RESP) | (1 << WLAN_FC_STYPE_AUTH);
	if (esp_wifi_register_mgmt_frame_internal(wpa_s->type, wpa_s->subtype) != ESP_OK) {
		ret = -1;
		goto err;
	}
	wpa_cb->wpa_sta_rx_mgmt = ieee80211_handle_rx_frm;
	return 0;
err:
	esp_supplicant_common_deinit();
	return ret;
}

void esp_supplicant_common_deinit(void)
{
	struct wpa_supplicant *wpa_s = &g_wpa_supp;

#if defined(CONFIG_WPA_11KV_SUPPORT)
	esp_scan_deinit(wpa_s);
	wpas_rrm_reset(wpa_s);
	wpas_clear_beacon_rep_data(wpa_s);
#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */
	if (wpa_s->type) {
		wpa_s->type = 0;
		esp_wifi_register_mgmt_frame_internal(wpa_s->type, wpa_s->subtype);
	}
#if defined(CONFIG_WPA_11KV_SUPPORT)
	if (!s_supplicant_task_hdl) {
	/*We have failed to create a task, delete queue and exit*/
		if (s_supplicant_evt_queue) {
			vQueueDelete(s_supplicant_evt_queue);
			s_supplicant_evt_queue = NULL;
		}
	}else if (esp_supplicant_post_evt(SIG_SUPPLICANT_DEL_TASK, 0) != 0) {
	/*Failed to post delete event, just delete the event queue and exit*/
		if (s_supplicant_evt_queue) {
			vQueueDelete(s_supplicant_evt_queue);
			s_supplicant_evt_queue = NULL;
		}
		wpa_printf(MSG_ERROR, "failed to send task delete event");
	}
	s_supplicant_task_init_done = false;
#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */
}

void supplicant_sta_conn_handler(uint8_t *bssid)
{
#if defined(CONFIG_WPA_11KV_SUPPORT)
	u8 *ie;
	struct wpa_supplicant *wpa_s = &g_wpa_supp;
	struct wpa_bss *bss = wpa_bss_get_bssid(wpa_s, bssid);
	if (!bss) {
		wpa_printf(MSG_INFO, "connected bss entry not present in scan cache");
		return;
	}
	wpa_s->current_bss = bss;
	ie = (u8 *)bss;
	ie += sizeof(struct wpa_bss);
	ieee802_11_parse_elems(wpa_s, ie, bss->ie_len);
	wpa_bss_flush(wpa_s);
	/* Register for mgmt frames */
	register_action_frame(wpa_s);
	/* clear set bssid flag */
	clear_bssid_flag(wpa_s);
#endif /* defined(CONFIG_WPA_11KV_SUPPORT)*/
}

void supplicant_sta_disconn_handler(void)
{
#if defined(CONFIG_WPA_11KV_SUPPORT)
	struct wpa_supplicant *wpa_s = &g_wpa_supp;

	wpas_rrm_reset(wpa_s);
	if (wpa_s->current_bss) {
		wpa_s->current_bss = NULL;
	}
	clear_bssid_flag(wpa_s);
#endif /* defined(CONFIG_WPA_11KV_SUPPORT) */
}

#if defined(CONFIG_WPA_11KV_SUPPORT)
bool esp_rrm_is_rrm_supported_connection(void)
{
	struct wpa_supplicant *wpa_s = &g_wpa_supp;

	if (!wpa_s->current_bss) {
		wpa_printf(MSG_DEBUG, "STA not associated, return");
		return false;
	}

	if (!(wpa_s->rrm_ie[0] & WLAN_RRM_CAPS_NEIGHBOR_REPORT)) {
		wpa_printf(MSG_DEBUG,
			"RRM: No network support for Neighbor Report.");
		return false;
	}

	return true;
}

int esp_rrm_send_neighbor_rep_request(neighbor_rep_request_cb cb,
				      void *cb_ctx)
{
	struct wpa_supplicant *wpa_s = &g_wpa_supp;
	struct wpa_ssid_value wpa_ssid = {0};
	struct wifi_ssid *ssid;

	if (!wpa_s->current_bss) {
		wpa_printf(MSG_ERROR, "STA not associated, return");
		return -2;
	}

	if (!(wpa_s->rrm_ie[0] & WLAN_RRM_CAPS_NEIGHBOR_REPORT)) {
		wpa_printf(MSG_ERROR,
			"RRM: No network support for Neighbor Report.");
		return -1;
	}

	ssid = esp_wifi_sta_get_prof_ssid_internal();

	os_memcpy(wpa_ssid.ssid, ssid->ssid, ssid->len);
	wpa_ssid.ssid_len = ssid->len;

	return wpas_rrm_send_neighbor_rep_request(wpa_s, &wpa_ssid, 0, 0, cb, cb_ctx);
}

bool esp_wnm_is_btm_supported_connection(void)
{
	struct wpa_supplicant *wpa_s = &g_wpa_supp;

	if (!wpa_s->current_bss) {
		wpa_printf(MSG_DEBUG, "STA not associated, return");
		return false;
	}

	if (!wpa_bss_ext_capab(wpa_s->current_bss, WLAN_EXT_CAPAB_BSS_TRANSITION)) {
		wpa_printf(MSG_DEBUG, "AP doesn't support BTM, return");
		return false;
	}

	return true;
}

int esp_wnm_send_bss_transition_mgmt_query(enum btm_query_reason query_reason,
					   const char *btm_candidates,
					   int cand_list)
{
	struct wpa_supplicant *wpa_s = &g_wpa_supp;

	if (!wpa_s->current_bss) {
		wpa_printf(MSG_ERROR, "STA not associated, return");
		return -2;
	}

	if (!wpa_bss_ext_capab(wpa_s->current_bss, WLAN_EXT_CAPAB_BSS_TRANSITION)) {
		wpa_printf(MSG_ERROR, "AP doesn't support BTM, return");
		return -1;
	}
	return wnm_send_bss_transition_mgmt_query(wpa_s, query_reason, btm_candidates, cand_list);
}

void wpa_supplicant_connect(struct wpa_supplicant *wpa_s,
			    struct wpa_bss *bss, char *ssid)
{
	wifi_config_t *config = os_zalloc(sizeof(wifi_config_t));

	if (!config) {
		wpa_printf(MSG_ERROR, "failed to allocate memory");
		return;
	}

	esp_wifi_get_config(WIFI_IF_STA, config);
	/* We only support roaming in same ESS, therefore only bssid setting is needed */
	os_memcpy(config->sta.bssid, bss->bssid, ETH_ALEN);
	config->sta.bssid_set = 1;
	esp_wifi_internal_issue_disconnect(WIFI_REASON_ROAMING);
	esp_wifi_set_config(WIFI_IF_STA, config);
	os_free(config);
	esp_wifi_connect();
}

void esp_set_rm_enabled_ie(void)
{
	uint8_t rmm_ie[5] = {0};
	uint8_t rrm_ie_len = 5;
	uint8_t *pos = rmm_ie;

	*pos |= WLAN_RRM_CAPS_LINK_MEASUREMENT;

	*pos |= WLAN_RRM_CAPS_BEACON_REPORT_PASSIVE |
#ifdef SCAN_CACHE_SUPPORTED
		WLAN_RRM_CAPS_BEACON_REPORT_TABLE |
#endif
		WLAN_RRM_CAPS_BEACON_REPORT_ACTIVE;

	/* set rm enabled IE if enabled in driver */
	if (esp_wifi_is_rm_enabled_internal(WIFI_IF_STA)) {
		esp_wifi_set_appie_internal(WIFI_APPIE_RM_ENABLED_CAPS, rmm_ie, rrm_ie_len, 0);
	}
}

static size_t get_rm_enabled_ie(uint8_t *ie, size_t len)
{
	uint8_t rrm_ie[7] = {0};
	uint8_t rrm_ie_len = 5;
	uint8_t *pos = rrm_ie;

	if (!esp_wifi_is_rm_enabled_internal(WIFI_IF_STA)) {
		return 0;
	}

	*pos++ = WLAN_EID_RRM_ENABLED_CAPABILITIES;
	*pos++ = rrm_ie_len;
	*pos |= WLAN_RRM_CAPS_LINK_MEASUREMENT;

	*pos |= WLAN_RRM_CAPS_BEACON_REPORT_PASSIVE |
#ifdef SCAN_CACHE_SUPPORTED
		WLAN_RRM_CAPS_BEACON_REPORT_TABLE |
#endif
		WLAN_RRM_CAPS_BEACON_REPORT_ACTIVE;

	os_memcpy(ie, rrm_ie, sizeof(rrm_ie));

	return rrm_ie_len + 2;
}
#endif

static uint8_t get_extended_caps_ie(uint8_t *ie, size_t len)
{
	uint8_t ext_caps_ie[5] = {0};
	uint8_t ext_caps_ie_len = 3;
	uint8_t *pos = ext_caps_ie;
	wifi_ioctl_config_t cfg = {0};

	if (!esp_wifi_is_btm_enabled_internal(WIFI_IF_STA)) {
		return 0;
	}

	*pos++ = WLAN_EID_EXT_CAPAB;
	*pos++ = ext_caps_ie_len;
	esp_err_t err = esp_wifi_internal_ioctl(WIFI_IOCTL_GET_STA_HT2040_COEX, &cfg);
	if (err == ESP_OK && cfg.data.ht2040_coex.enable) {
		*pos++ |= BIT(WLAN_EXT_CAPAB_20_40_COEX);
	} else {
		*pos++ = 0;
    }
	*pos++ = 0;
#define CAPAB_BSS_TRANSITION BIT(3)
	*pos |= CAPAB_BSS_TRANSITION;
#undef CAPAB_BSS_TRANSITION
	os_memcpy(ie, ext_caps_ie, sizeof(ext_caps_ie));

	return ext_caps_ie_len + 2;
}

void esp_set_assoc_ie(uint8_t *bssid, const u8 *ies, size_t ies_len, bool mdie)
{
#define ASSOC_IE_LEN 128
	uint8_t *ie, *pos;
	size_t len = ASSOC_IE_LEN, ie_len;
	ie = os_malloc(ASSOC_IE_LEN + ies_len);
	if (!ie) {
		wpa_printf(MSG_ERROR, "failed to allocate ie");
		return;
	}
	pos = ie;
	ie_len = get_extended_caps_ie(pos, len);
	pos += ie_len;
	len -= ie_len;
#ifdef CONFIG_WPA_11KV_SUPPORT
	ie_len = get_rm_enabled_ie(pos, len);
	pos += ie_len;
	len -= ie_len;
#ifdef CONFIG_MBO
	ie_len = get_operating_class_ie(pos, len);
	pos += ie_len;
	len -= ie_len;
	ie_len = get_mbo_oce_assoc_ie(pos, len);
	pos += ie_len;
	len -= ie_len;
#endif /* CONFIG_MBO */
#endif
	if (ies_len) {
		os_memcpy(pos, ies, ies_len);
		pos += ies_len;
		len -= ies_len;
	}
	esp_wifi_unset_appie_internal(WIFI_APPIE_ASSOC_REQ);
	esp_wifi_set_appie_internal(WIFI_APPIE_ASSOC_REQ, ie, ASSOC_IE_LEN - len, 0);
	os_free(ie);
#undef ASSOC_IE_LEN
}

#ifdef CONFIG_WPA_11KV_SUPPORT
void esp_get_tx_power(uint8_t *tx_power)
{
#define DEFAULT_MAX_TX_POWER 19 /* max tx power is 19.5 dbm */
	s8 power;
	/* esp sends management frames at max tx power configured */
	int ret = esp_wifi_get_max_tx_power(&power);
	if (ret != 0) {
		wpa_printf(MSG_ERROR, "failed to get tx power");
		*tx_power = DEFAULT_MAX_TX_POWER;
		return;
	}
	*tx_power = power/4;
#undef DEFAULT_MAX_TX_POWER
}

int wpa_drv_send_action(struct wpa_supplicant *wpa_s,
			unsigned int channel,
			unsigned int wait,
			const u8 *data, size_t data_len,
			int no_cck)
{
	int ret = 0;
	wifi_mgmt_frm_req_t *req = os_zalloc(sizeof(*req) + data_len);;
	if (!req)
		return -1;

	if (!wpa_s->current_bss) {
		wpa_printf(MSG_ERROR, "STA not associated, return");
		ret = -1;
		goto cleanup;
	}

	req->ifx = WIFI_IF_STA;
	req->subtype = WLAN_FC_STYPE_ACTION;
	req->data_len = data_len;
	os_memcpy(req->data, data, req->data_len);

	if (esp_wifi_send_mgmt_frm_internal(req) != 0) {
		wpa_printf(MSG_ERROR, "action frame sending failed");
		ret = -1;
		goto cleanup;
	}
	wpa_printf(MSG_INFO, "action frame sent");

cleanup:
	os_free(req);
	return ret;
}

int esp_supplicant_post_evt(uint32_t evt_id, uint32_t data)
{
	supplicant_event_t *evt = os_zalloc(sizeof(supplicant_event_t));
	if (!evt) {
		wpa_printf(MSG_ERROR, "Failed to allocate memory.");
		return -1;
	}
	evt->id = evt_id;
	evt->data = data;

	SUPPLICANT_API_LOCK();
	/*Make sure no event can be sent when deletion event is sent or the task is not initialized*/
	if (!s_supplicant_task_init_done) {
		SUPPLICANT_API_UNLOCK();
		os_free(evt);
		return -1;
	}
	if (xQueueSend(s_supplicant_evt_queue, &evt, 10 / portTICK_PERIOD_MS ) != pdPASS) {
		SUPPLICANT_API_UNLOCK();
		os_free(evt);
		return -1;
	}
	if (evt_id == SIG_SUPPLICANT_DEL_TASK) {
		s_supplicant_task_init_done = false;
	}
	SUPPLICANT_API_UNLOCK();
	return 0;
}
#endif
