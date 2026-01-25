import subprocess
import time
import sys
import logging
import argparse
import os
from datetime import datetime

# --- CONFIG ---
WATCHER_LOG = 'watcher.log'
DEFAULT_COOLDOWN = 5 # Seconds
MAX_COOLDOWN = 600 # 10 Minutes

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] WATCHER: %(message)s',
    handlers=[
        logging.FileHandler(WATCHER_LOG, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class ProcessWatcher:
    def __init__(self, target_script, interpreter='python'):
        self.target_script = target_script
        self.interpreter = interpreter
        self.restart_count = 0
        self.last_restart_time = time.time()
        self.current_cooldown = DEFAULT_COOLDOWN

    def run(self):
        if not os.path.exists(self.target_script):
            logging.error(f"❌ Target script not found: {self.target_script}")
            return

        logging.info(f"🛡️ STARTING WATCHER for: {self.target_script}")
        logging.info("Press Ctrl+C to stop the watcher and the child process.")

        while True:
            try:
                # Start Process
                logging.info(f"🚀 Launching process #{self.restart_count + 1}...")
                
                start_time = time.time()
                
                # Using shell=False for better signal handling usually, but shell=True works well on Windows for batch
                # Here we call python directly so shell=False is cleaner
                process = subprocess.Popen([self.interpreter, self.target_script])
                
                # Wait for completion
                return_code = process.wait()
                
                duration = time.time() - start_time
                
                # Check Exit Status
                if return_code == 0:
                    logging.info(f"✅ Process finished successfully (Duration: {duration:.1f}s). Restarting...")
                    self.current_cooldown = DEFAULT_COOLDOWN # Reset cooldown on clean exit
                else:
                    logging.warning(f"⚠️ Process CRASHED with code {return_code} (Duration: {duration:.1f}s)")
                    
                    # Backoff Logic
                    # If it crashed quickly (< 60s), double the cooldown
                    if duration < 60:
                        self.current_cooldown = min(self.current_cooldown * 2, MAX_COOLDOWN)
                        logging.warning(f"📉 Quick Crash Detected. Increasing cooldown to {self.current_cooldown}s")
                    else:
                        # If it ran for a while, reset cooldown
                        self.current_cooldown = DEFAULT_COOLDOWN
                        logging.info("⏱️ Long run detected. Resetting cooldown.")

                self.restart_count += 1
                
                # Wait before restart
                logging.info(f"💤 Waiting {self.current_cooldown}s before restart...")
                time.sleep(self.current_cooldown)

            except KeyboardInterrupt:
                logging.info("\n🛑 WATCHER STOPPING: User interrupted.")
                # Kill child if still alive (wait() handles this mostly, but if we are in sleep)
                if 'process' in locals() and process.poll() is None:
                    logging.info("Killing active process...")
                    process.terminate()
                break
            except Exception as e:
                logging.critical(f"🔥 WATCHER ERROR: {e}")
                time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holonic System Watcher")
    parser.add_argument('--target', type=str, required=True, help="Path to the python script to watch")
    parser.add_argument('--interpreter', type=str, default='python', help="Python interpreter path/command")
    
    args = parser.parse_args()
    
    watcher = ProcessWatcher(args.target, args.interpreter)
    watcher.run()
