import os, sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

import shutil

import smtplib

os.chdir('/root/Novosad/mouses/')
password = "Novosad1441262"
emails = ['victor@pine.bio', 'elia@pine.bio']


def send_mail(message, img_dir=None):
    msg = MIMEMultipart()

    msg['From'] = "victor@pine.bio"
    msg['Subject'] = "Mouse Tracker"

    if img_dir is not None:
        shutil.make_archive(img_dir, 'zip', '/'.join(img_dir.split('/')[:-1]), img_dir.split('/')[-1])

        part = MIMEBase("application", "octet-stream")
        part.set_payload(open(img_dir + ".zip", "rb").read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=\"%s.zip\"" % (img_dir.split('/')[-1]))
        msg.attach(part)
        msg.attach(MIMEText(message, 'plain'))

    # create server
    server = smtplib.SMTP('smtp.gmail.com: 587')

    server.starttls()

    # Login Credentials for sending the mail
    server.login(msg['From'], password)

    # send the message via the server.
    for email in emails[:-1]:
        msg['To'] = email
        server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()
    print('email {} sended'.format(message))

    if img_dir is not None:
        os.remove(img_dir + '.zip')


if __name__ == '__main__':
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]

    send_mail('Hello just checking', path)