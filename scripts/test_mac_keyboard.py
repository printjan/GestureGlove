#!/usr/bin/env python
import time
import sys

# Ensure pyautogui is installed or importable
try:
    import pyautogui
except ImportError:
    print("Error: pyautogui is not installed in the current environment.")
    print("Please install it with: pip install pyautogui")
    sys.exit(1)

def test_pyautogui_methods():
    print("=" * 80)
    print("             macOS PyAutoGUI Keybinding Test Script for PowerPoint")
    print("=" * 80)
    print("This script will test 4 different methods of sending 'Command + Shift + Return'.")
    print("Please switch focus to your PowerPoint window immediately after choosing a method.")
    print("-" * 80)
    print("1. Method A: Standard pyautogui.hotkey('win', 'shift', 'return')")
    print("2. Method B: pyautogui.hotkey('command', 'shift', 'return') - using 'command' directly")
    print("3. Method C: Standard pyautogui.hotkey with a 100ms interval between keys")
    print("4. Method D: Manual keyDown/keyUp sequence with 100ms sleeps (most reliable for macOS OS events)")
    print("5. Method E: Send 'command+return' (Start from current slide) to verify basic Return key triggers")
    print("6. Run all methods with 5 second delays in between")
    print("7. Exit")
    print("-" * 80)
    
    try:
        choice = input("Enter choice (1-7): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if choice == "7":
        sys.exit(0)

    def run_method_a():
        print("\n[Method A] Sending: hotkey('win', 'shift', 'return') in 3 seconds...")
        time.sleep(3)
        pyautogui.hotkey('win', 'shift', 'return')
        print("Sent!")

    def run_method_b():
        print("\n[Method B] Sending: hotkey('command', 'shift', 'return') in 3 seconds...")
        time.sleep(3)
        pyautogui.hotkey('command', 'shift', 'return')
        print("Sent!")

    def run_method_c():
        print("\n[Method C] Sending: hotkey('command', 'shift', 'return', interval=0.1) in 3 seconds...")
        time.sleep(3)
        pyautogui.hotkey('command', 'shift', 'return', interval=0.1)
        print("Sent!")

    def run_method_d():
        print("\n[Method D] Sending manual keyDown/keyUp sequence with 100ms delays in 3 seconds...")
        time.sleep(3)
        print("Pressing command...")
        pyautogui.keyDown('command')
        time.sleep(0.1)
        print("Pressing shift...")
        pyautogui.keyDown('shift')
        time.sleep(0.1)
        print("Pressing return...")
        pyautogui.press('return')
        time.sleep(0.1)
        print("Releasing shift...")
        pyautogui.keyUp('shift')
        time.sleep(0.1)
        print("Releasing command...")
        pyautogui.keyUp('command')
        print("Sequence complete!")

    def run_method_e():
        print("\n[Method E] Sending: hotkey('command', 'return') in 3 seconds...")
        time.sleep(3)
        pyautogui.hotkey('command', 'return')
        print("Sent!")

    if choice == "1":
        run_method_a()
    elif choice == "2":
        run_method_b()
    elif choice == "3":
        run_method_c()
    elif choice == "4":
        run_method_d()
    elif choice == "5":
        run_method_e()
    elif choice == "6":
        run_method_a()
        time.sleep(5)
        run_method_b()
        time.sleep(5)
        run_method_c()
        time.sleep(5)
        run_method_d()
        time.sleep(5)
        run_method_e()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    test_pyautogui_methods()
