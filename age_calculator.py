import datetime

def calculate_age(birthdate):
    today = datetime.date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

def main():
    # User inputs their date of birth
    date_of_birth = input("Enter your date of birth (YYYY-MM-DD): ")
    birthdate = datetime.datetime.strptime(date_of_birth, '%Y-%m-%d').date()
    
    # Calculate age
    age = calculate_age(birthdate)
    
    # Print the age
    print("You are {} years old.".format(age))

if __name__ == "__main__":
    main()
