from Tkinter import *
from threading import Thread
from os.path import expanduser
import os
import time
import datetime
import tkFont

def recThread():
        os.system("sleep 1s;ffmpeg -f x11grab -s $(xdpyinfo | grep 'dimensions:' | awk '{print $2}' | cut -dx -f1)x$(xdpyinfo | grep 'dimensions:' | awk '{print $2}' | cut -dx -f2) -b:v 350k -r 15 -i :0.0  -q:v 3 -r 15 -y ~/Videos/$(date +%d%b_%Hh%Mm).avi &")
def rec():
        global videoFile
        mydate = datetime.datetime.now()
        videoFile = mydate.strftime("%d%b_%Hh%Mm.avi")
        pathSt=os.getcwd()+"/Videos/"
        l['text']=os.path.expanduser('~')+"/Videos/"
        l1['text']=videoFile
        b.config(state=DISABLED)
        b1.config(state=ACTIVE)
        t = Thread(target=recThread)
        t.start()
        global count_flag, secs, mins
        count_flag = True
        secs=0
        mins=0
        while True:
                if count_flag == False:
                        break
                label['text'] = str("%02dm:%02ds" % (mins,secs))
                if secs == 0:
                        time.sleep(0)
                else:
                        time.sleep(1)
                if(mins==0 and secs==1):
                        b1.config(bg="red")
                        b.config(fg="white")
                        b.config(bg="white")
                if secs==60:
                        secs=0
                        mins+=1
                        label['text'] = str("%02dm:%02ds" % (mins,secs))
                root.update()
                secs = secs+1
def stop():

        b.config(state=ACTIVE)
        b1.config(state=DISABLED)
        b1.config(fg="white")
        b1.config(bg="white")
        b.config(fg="white")
        b.config(bg="green")
        global count_flag
        count_flag = False
        os.system("pkill -n ffmpeg")
        try:
            t.stop()
        except:
            print("")

root = Tk()
fontTime = tkFont.Font(family="Helvetica", size=12)
fontButton = tkFont.Font(family="Monospace", size=11,weight="bold")
label = Label(root, text="00m:00s",fg="blue",font="fontTime")
b = Button(root,text="Record",command=rec,state=ACTIVE,bg="green",font="fontButton")
b1 = Button(root,text=" Stop ",command=stop,state=DISABLED,bg="white",font="fontButton")
l = Label(root, text="")
l1 = Label(root, text="")
label.grid(row=0, column=0, columnspan=2)
b.grid(row=1, column=0, padx=1, pady=5)
b1.grid(row=1, column=1, padx=1)
l.grid(row=2, column=0,columnspan=2)
l1.grid(row=3, column=0,columnspan=2)
root.minsize(160,105)
root.maxsize(160,105)
root.title("Desktop REC")
root.attributes("-topmost", 1)
root.mainloop()