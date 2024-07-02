| Supported Targets | ESP32-S2 |
| ----------------- | -------- |

# TinyUSB Sample Descriptor

(See the README.md file in the upper level 'examples' directory for more information about examples.)

This example is demonstrating how to set up ESP32-S2 chip to work as a Generic USB Device with a user-defined descriptor. You can specify a manufacturer, device's name, ID and other USB-devices parameters responsible for identification by host.

As a USB stack, a TinyUSB component is used.

## How to use example

### Hardware Required

- Any board with the ESP32-S2 chip with USB connectors or with exposed USB's D+ and D- (DATA+/DATA-) pins.

If the board has no USB connector, but has the pins connect pins directly to the host (e.g. with DIY cable from any USB connection cable)

```
ESP32-S2 BOARD          USB CONNECTOR (type A)
                          --
                         | || VCC
    [GPIO 19]  --------> | || D-
    [GPIO 20]  --------> | || D+
                         | || GND
                          --
```

You can also use power from the USB connector.

### Configure the project

There are two ways to set up a descriptor - using Menuconfig tool and in-code

#### In-code setting up

For the manual descriptor's configuration use the default example's settings and modify `tusb_sample_descriptor.c` according to your needs

#### Menuconfig

If you want to set up the descriptor using Menuconfig UI:

1. Execute in the terminal from the example's directory: `idf.py menuconfig`

2. Turn off `Set up a USB descriptor manually in code` parameter at `Example Configuration`

3. Follow `Component config -> TinyUSB -> Descriptor configuration` for all available configurations.

### Build and Flash

Build the project and flash it to the board, then run monitor tool to view serial output:

```bash
idf.py -p PORT flash monitor
```

(Replace PORT with the name of the serial port to use.)

(To exit the serial monitor, type ``Ctrl-]``.)

See the Getting Started Guide for full steps to configure and use ESP-IDF to build projects.

## Example Output

After the flashing you should see the output:

```
I (349) TinyUSB: Driver installation...
I (349) TinyUSB - Descriptors Control: Setting of a descriptor:
.bDeviceClass       = 0
.bDeviceSubClass    = 0,
.bDeviceProtocol    = 0,
.bMaxPacketSize0    = 64,
.idVendor           = 0x0000303a,
.idProduct          = 0x00003000,
.bcdDevice          = 0x00000101,
.iManufacturer      = 0x01,
.iProduct           = 0x02,
.iSerialNumber      = 0x03,
.bNumConfigurations = 0x01

I (389) TinyUSB: Driver installed
I (389) example: USB initialization DONE
```
