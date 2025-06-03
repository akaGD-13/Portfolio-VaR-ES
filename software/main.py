# main.py
import sys

# historical_calibration.py and input_mu_sigma.py should be in the same directory or on PYTHONPATH
import historical_calibration
import input_mu_sigma

def main():
    print("Choose input mode:")
    print("  1) Historical calibration from CSV prices (includes historical VaR and ES)")
    print("  2) Manual parameter input (no historical data needed)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        historical_calibration.main()
    elif choice == '2':
        input_mu_sigma.main()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
        sys.exit(1)

if __name__ == '__main__':
    main()
