import subprocess
import sys

# Paths to the scripts
record_both_script = 'data_capture/record_both.py'
audio_plot_script = 'data_capture/audio_plot.py'
csi_plot_script = 'data_capture/csi_plot.py'
audio_matrix_script = 'data_capture/wav_to_matrix.py'
csi_matrix_script = 'data_capture/csi_to_matrix.py'

def run_script(script_path):
    """Run a script and wait for it to complete."""
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(f"Output of {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:\n{e.stderr}")
        # sys.exit(1)

def main():
    print(f"Running {record_both_script}...")
    run_script(record_both_script)

    print(f"Running {audio_plot_script}...")
    run_script(audio_plot_script)

    print(f"Running {csi_plot_script}...")
    run_script(csi_plot_script)

    print(f"Running {audio_matrix_script}...")
    run_script(audio_matrix_script)

    print(f"Running {csi_matrix_script}...")
    run_script(csi_matrix_script)

if __name__ == "__main__":
    main()
