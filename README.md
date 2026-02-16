Face Attendance System (Offline) — Desktop App

Tech:
- Python + Tkinter (desktop UI)
- OpenCV YuNet (face detector) + SFace (face recognizer)
- Offline operation (no internet required after dependencies/models are present)

Features:
- Enrollment: capture multiple samples for each student (Student ID + Full Name + Class)
- Attendance: recognizes students from the camera and records attendance (cooldown prevents duplicates)
- Admin panel: manage classes, settings, and enrolled students
- Fullscreen mode with admin-password exit

Default admin password: 987887787

Run on Windows:
1) Extract the ZIP
2) Double-click: run_app.bat

Notes:
- For best accuracy, enroll students using the webcam in similar lighting to the classroom.
- Camera must be accessible to Windows (check Privacy settings).
