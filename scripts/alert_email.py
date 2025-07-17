# scripts/alert_email.py

import smtplib
from email.mime.text import MIMEText

def send_fraud_alert(probability):
    sender = "tirthparmar1507@gmail.com"
    password = "myphlsfztgsnjquy"  # ⚠️ App password (not Gmail password)
    recipient = "tirthparmar1507@gmail.com"  # or any recipient

    subject = "🚨 Fraudulent Transaction Detected!"
    body = f"A transaction has been flagged as fraud with {probability:.2f}% confidence."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            print("✅ Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)
