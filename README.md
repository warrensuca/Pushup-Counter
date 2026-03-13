<h1>Pushup Counter</h1> 
A real-time pushup counter using your webcam, MediaPipe pose detection, and OpenCV.

<h1>Requirements to run locally</h1>
<ol>
  <li>A webcam</li>
  <li>Python 3</li>
  <li>Allow VS Code access to your webcam</li>
  <li><code>pip install opencv-python mediapipe numpy</code></li>
  
</ol>

<h1>Purpose</h1>
I created this project for practice with Computer Vision, and a tune up for a future project that involves movement!

<h1>Methodology</h1>

I used Google's MediaPipe as well as OpenCV to track 33 points of the body. The body parts I examined to register a pushup were the 
shoulder and elbow. Originally, I said if the shoulder is below the elbow, it is a pushup, but made changes to be more lenient. I then
chose to turn the detection points green for a visual cue that a successful pushup was performed. 

<h1>Notes</h1>
<ol>
  <li>The camera window is an OpenCV window and does not have a native macOS close button — use Q to exit</li>
  <li><b>Place your camera head on for best angle detection</b></li>
</ol>
