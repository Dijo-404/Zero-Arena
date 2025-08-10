**How to Run the Application**

**Step 1 — Install Requirements**
It is recommended to use a Conda environment to avoid dependency conflicts.
## Create and activate a new conda environment (optional but recommended)
```bash
conda create -n myenv python=3.10
conda activate myenv
```

## Install dependencies
```bash
pip install -r requirements.txt
Step 2 — Run the Application
Start the Flask application and log its output to a file:
python app.py > flask_output.log &
```
This runs the application in the background and stores the console output in flask_output.log.

**Step 3 — Access the Camera Feed**
Once the application is running, open the following URL in your browser:
```
http://127.0.0.1:5000/
```
You should now be able to view the live camera feed from the application.

**Notes**
-Ensure that your camera is connected and accessible.
-If running on a remote server, replace 127.0.0.1 with the server’s IP address.
-To stop the application running in the background, find its process ID:
```
ps aux | grep app.py
```
and terminate it.

