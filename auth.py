import streamlit as st

# Function to authenticate user
def authenticate_user(username, password):
    # Your authentication logic goes here
    # For demonstration purposes, return True if username and password match
    return username == "admin" and password == "password"

# Streamlit UI for login page
def login():
    st.title('Login')

    # Login form
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if authenticate_user(username, password):
            st.success('Login successful!')
            # Redirect to another page after login
            st.experimental_rerun()
        else:
            st.error('Invalid username or password')

# Streamlit UI for registration page
def registration():
    st.title('Registration')

    # Registration form
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    confirm_password = st.text_input('Confirm Password', type='password')

    if st.button('Register'):
        # Your registration logic goes here
        # For demonstration purposes, print the registered username and password
        if password == confirm_password:
            st.success(f'Registration successful! Username: {username}, Password: {password}')
        else:
            st.error('Passwords do not match')

# Main function
def main():
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio("Go to", ('Login', 'Registration'))

    if choice == 'Login':
        login()
    elif choice == 'Registration':
        registration()

if __name__ == '__main__':
    main()
