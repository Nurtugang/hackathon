import smtplib
import ssl

def send_email(email_to, message):
    smtp_port = 587                
    smtp_server = "smtp.gmail.com"  
    email_from = "nureek2001@gmail.com"
    pswd = "ysgzqiazrouqxqal"
    simple_email_context = ssl.create_default_context()
    try:
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls(context=simple_email_context)
        TIE_server.login(email_from, pswd)
        print("Успешное подключение к smtp")
        
        print()
        TIE_server.sendmail(email_from, email_to, message)
        print(f" Сообщение успешно отправлено к {email_to}")
    except Exception as e:
        print(e)
    finally:
        TIE_server.quit()




