import os
import sys

def reset_monitor():
    path = os.path.join(os.getcwd(), 'monitor_state.json')
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"✅ SUCCESSFULLY DELETED: {path}")
            print("The System Monitor will now treat the next run as a Fresh Start.")
        except Exception as e:
            print(f"❌ ERROR: Could not delete file: {e}")
    else:
        print(f"ℹ️ No monitor state found at: {path}")
        print("System is already clean.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to reset the Monitor Health State? (type 'yes'): ")
    if confirm.lower() == 'yes':
        reset_monitor()
    else:
        print("Operation Cancelled.")
