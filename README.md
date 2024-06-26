**Content Filter Extension Backend**

This is the backend component of the Content Filter Extension project, built using Flask. This project aims to provide the necessary functionality for filtering content based on predefined criteria.
This project is made handle the requests from our chrome extension projcet (content filter) [https://github.com/Andrew-A-A/Contient_Filter_chrome_Extenstion]

### Setting Up Virtual Environment

To ensure a clean and isolated environment for running this Flask project, it is recommended to use a virtual environment. Follow these steps to set it up:

1. **Clone the Repository:**
   ```
   git clone https://github.com/Andrew-A-A/Content_Filter_Extension_Backend.git
   ```

2. **Navigate to the Project Directory:**
   ```
   cd Content_Filter_Extension_Backend
   ```

3. **Create a Virtual Environment:**
   ```
   python -m venv venv
   ```

4. **Activate the Virtual Environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```

### Installing Required Packages

This project manages its dependencies using a `requirements.txt` file. Once you've activated your virtual environment, you can install the required packages using pip:

```
pip install -r requirements.txt
```

### Running the Application

After setting up the virtual environment and installing the necessary packages, you can run the Flask application. By default, the Flask development server will run on `http://127.0.0.1:5000/`.

- To start the server, run the following command:

```
flask --app app run
```
- To start in debug mode:

```
flask --app app run --debug 
```
