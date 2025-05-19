import sys

def print_menu():
    print("\n🧠 NLP Assignment 2025")
    print("1) Τρέξε Reconstruction Pipelines (Παραδοτέο 1)")
    print("2) Τρέξε Υπολογιστική Ανάλυση (Παραδοτέο 2)")
    print("3) Έξοδος")

def main_menu():
    while True:
        print_menu()
        choice = input("👉 Επιλογή (1/2/3): ")

        if choice == "1":
            from nlp_assignment_2025.main import main as reconstruction_main
            reconstruction_main()
        elif choice == "2":
            from nlp_assignment_2025.analysis_main import main as analysis_main
            analysis_main()
        elif choice == "3":
            print("👋 Έξοδος.")
            sys.exit(0)
        else:
            print("⛔ Μη έγκυρη επιλογή. Δοκίμασε ξανά.")

if __name__ == "__main__":
    main_menu()