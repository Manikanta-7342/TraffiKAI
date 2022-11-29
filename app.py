import streamlit as st
import os,tempfile
from pathlib import Path
import pandas as pd
import threading,time
from PIL import Image

st.set_page_config(page_title="TraffiKAI", page_icon="./icon.jpg", layout="wide", initial_sidebar_state="auto", menu_items=None)

m1,m2,_=st.columns((0.5,1,4))
with m1:
    image = Image.open("icon.jpg")
    st.image(image)
with m2:
    image = Image.open("title.png")
    st.image(image)
    # st.title("TraffikAI")
    # st.write("A-EYE on ROADS")

#st.markdown("----", unsafe_allow_html=True)
def get_path(file_name):
    li = file_name.split("\\")
    li.insert(3,'Test')
    return "\\\\".join(li)

def violation():
    os.system("python .\Violation-Detection-System\\violation.py " )
def emergecy_result():
    file_emer = open('emer.txt', 'r+')
    file_read = file_emer.read()
    emer = file_read[1:-1]
    emer = [int(x) for x in emer.split(",")]
    cont_1 = emer.index(1) if 1 in emer else 'None'
    return cont_1

def emergency(east,south,west,north):
    os.system('python emergency.py '+east+' '+south+' '+west+' '+north)

def ssd(east,south,west,north):
    os.system('python ssd_script.py '+east+' '+south+' '+west+' '+north)

def gui():
    os.system('python work2.py')



def execute(east_path,south_path, west_path, north_path):

    li = (east_path, south_path, west_path, north_path)
    t1 = threading.Thread(target=emergency, args=li)
    t1.start()
    with st.spinner('Wait for it...'):
        my_bar = st.progress(0)
        with st.empty():
            for percent_complete in range(100):
                time.sleep(1)
                st.write(percent_complete)
                my_bar.progress(percent_complete + 1)
                if percent_complete == 70:
                    t2 = threading.Thread(target=ssd, args=li)
                    t2.start()
    # time.sleep(10)
    t1.join()
    t3 = threading.Thread(target=gui)
    t3.start()

def cases(case):
    if case==1:
        east_path="Test\\test1\\1080p\\7.0.1080.mp4"
        south_path="Test\\test1\\1080p\\3.2.1080.mp4"
        west_path="Test\\test1\\Emergency\\6.0.720.mp4"
        north_path="Test\\test1\\480p\\2.0.480.mp4"
    elif case==2:
        east_path="Test\\test1\\720p\\9.2.720.mp4"
        south_path="Test\\test1\\480p\\7.0.480.mp4"
        west_path="Test\\test1\\1080p\\3.2.1080.mp4"
        north_path="Test\\test1\\1080p\\16.0.1080.mp4"

    execute(east_path, south_path, west_path, north_path)
    return (east_path, south_path, west_path, north_path)

east_file = None
south_file = None
west_file = None
north_file = None

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            button[data-baseweb="tab"]{font-size:22px;}
            </style>
            """
st.markdown(hide_streamlit_style,unsafe_allow_html=True)
tabs = st.tabs(["Overview","Dynamic Traffic","Traffic Violation","Statistics"])
st.markdown("",unsafe_allow_html=False)
tab_ove=tabs[0]
tab_dyn=tabs[1]
tab_vio=tabs[2]
tab_sta=tabs[3]

with tab_ove:
    m1,m2=st.columns(2)
    with m1:
        st.header("Why TraffiKAI?")
        st.write("""Bengaluru is placed in 10th position for its global traffic index.The remarkable shift happened within two years when it ranked as the most congested city globally in 2019.
Bengaluru’s congestion levels came down to 48% in 2021.

From the TomTom(Tom2), the geolocation technology specialist, we acquired the last 7 days' congestion periods to be from 6 pm to 7 pm with congestion of 60 to 70 percent.

According to the National Crime Records Bureau, nearly 24,012 people die each day due to a delay in 
getting medical assistance. Many accident victims wait for help at the site, and a delay costs them their lives. The reasons could range from ambulances stuck in traffic to the fire brigade being unable to reach the site on time due to traffic jams.""")
        st.header("Density Algorithm")
        st.write("""The density score is calculated based on number of vehicles and considering each variant among them.
                Each vehicle has its own predefind density value.
                Formula, """)
        st.latex("Density = \sum_{k=0}^{n-1} axk/n")
        st.write("where xk = Different vehicles")
    with m2:
        im=Image.open('ov1.jpg')
        st.image(im)

with tab_dyn:
    d1,d2=st.columns(2)
    with d1:
        st.header("Assumptions:  ")
        st.write("""◉ The primary assumption of this model is that there will be no signal skipping and the starting point will be dependent on the lane in which the emergency vehicle is detected.\n
◉ If no emergency vehicle is detected, clockwise direction will be followed (starting from East).\n
◉ The videos are to be captured only for a single lane and should not overlap with the subsequent lanes for better accuracy in object detection.\n
◉ The videos are expected to have a fair amount of resolution for efficient results. (1080P (FHD) preferred)\n
◉ The installation of camera should be in respect to each particular lane.\n
""")
    with d2:
        st.header("Cost")
        st.write("""◉ Installation of coherent audio sensors at each signal to detect the emergency vehicles.\n
◉ High resolution cameras to be installed for better video capturing.\n
◉ Sufficient amount of processing power to run the model flawlessly.\n
""")

    l1,l2=st.columns(2)
    with l1:
        east_path, south_path, west_path, north_path = None, None, None, None
        st.subheader("Case 1:")
        st.text("East : High density South : Low density")
        st.text("West : Emergency    North : Low density")
        if st.button("Run Case 1"):
            tup=cases(1)
            east_path, south_path, west_path, north_path = tup[0],tup[1],tup[2],tup[3]
    with l2:
        st.subheader("Case 2:")
        st.text("East : High density South : High density")
        st.text("West : Low density North : High density")
        if st.button("Run Case 2"):
            tup = cases(2)
            east_path, south_path, west_path, north_path = tup[0], tup[1], tup[2], tup[3]
    st.markdown("----", unsafe_allow_html=True)
    st.subheader("Manual Demo:")
    c1,c2=st.columns(2)

    with c1:
        east_file = st.file_uploader("Upload East camera video", type='mp4')
        south_file = st.file_uploader("Upload South camera video", type='mp4')
    with c2:
        west_file = st.file_uploader("Upload West camera video", type='mp4')
        north_file = st.file_uploader("Upload North camera video", type='mp4')
    columns = st.columns((4.3, 1, 4.3))

    if columns[1].button('Run Model'):
        if (east_file == None or south_file == None or west_file == None or north_file == None):
            st.warning('Please Upload All Files', icon="⚠️")
        else:
                # time.sleep(5)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_1_file:
                # st.markdown("## Original video file")
                fp = Path(tmp_1_file.name)
                fp.write_bytes(east_file.getvalue())
                east_path = tmp_1_file.name
            with tempfile.NamedTemporaryFile(delete=False) as tmp_2_file:
                fp = Path(tmp_2_file.name)
                fp.write_bytes(south_file.getvalue())
                south_path = tmp_2_file.name

            with tempfile.NamedTemporaryFile(delete=False) as tmp_3_file:
                fp = Path(tmp_3_file.name)
                fp.write_bytes(west_file.getvalue())
                west_path = tmp_3_file.name

            with tempfile.NamedTemporaryFile(delete=False) as tmp_4_file:
                fp = Path(tmp_4_file.name)
                fp.write_bytes(north_file.getvalue())
                north_path = tmp_4_file.name
            execute(east_path, south_path, west_path, north_path)
    #time.sleep(10)
    try:
        cont=emergecy_result()
        with c1:
            video_file1 = open(east_path, 'rb')
            st.subheader("East Lane")
            video_bytes = video_file1.read()
            st.video(video_bytes)
            if cont==0:
                st.write("Emergency_Vehicle")
            else:
                st.write("Non Emergency_Vehicle")

            st.subheader("West Lane")
            video_file3 = open(west_path, 'rb')
            video_bytes = video_file3.read()
            st.video(video_bytes)
            if cont == 2:
                st.write("Emergency_Vehicle")
            else :
                st.write("Non Emergency_Vehicle")
        with c2:
            st.subheader("South Lane")
            video_file2 = open(south_path, 'rb')
            video_bytes = video_file2.read()
            st.video(video_bytes)
            if cont == 1:
                st.write("Emergency_Vehicle")
            else :
                st.write("Non Emergency_Vehicle")

            st.subheader("North Lane")
            video_file4 = open(north_path, 'rb')
            video_bytes = video_file4.read()
            st.video(video_bytes)
            if cont == 3:
                st.write("Emergency_Vehicle")
            else:
                st.write("Non Emergency_Vehicle")
    except:
        pass

with tab_vio:

    d1, d2 = st.columns(2)
    col = st.columns((4, 1, 4))
    with d1:
        st.header("Assumptions:  ")
        st.write("""◉ The primary assumption of this model is that the region of interest is set while the algorithm is installed in the end system.\n
◉ This model works only when the signal is red to detect any signal skipping violations occurring.\n
◉ The frame rate of the videos is to be set at the time of installation depending upon the scenario in a live traffic environment.\n
◉ The videos are expected to have a fair amount of resolution for efficient results. (1080P (FHD) preferred)\n

            """)
    with d2:
        st.header("Cost")
        st.write("""◉ Sufficient amount of processing power to run the model flawlessly.\n
◉ High resolution cameras to be installed for better video capturing.

            """)
    if col[1].button('Run Model',key=2):


        t4= threading.Thread(target=violation)
        t4.start()

        with st.spinner('Wait for it...'):
            my_bar = st.progress(0)
            with st.empty():
                for percent_complete in range(100):
                    time.sleep(0.13)
                    st.write(percent_complete)
                    my_bar.progress(percent_complete + 1)
                my_bar.progress(percent_complete + 1)

        t4.join()
        img_path = r"D:\Centuriton\models\Violation-Detection-System\Detected Images\\"
        i1, i2, i3 = st.columns(3)
        c = 0
        for i in os.listdir(img_path):
            image = Image.open(img_path + i)
            if c == 0:
                with i1:
                    st.image(image, caption=i, width=282)
                c += 1
            elif c == 1:
                with i2:
                    st.image(image, caption=i, width=282)
                c += 1
            else:
                with i3:
                    st.image(image, caption=i, width=282)
                c = 0

with tab_sta:
    _, row3_1,_ = st.columns((.2, 7.1, .2))
    data = pd.read_excel("D:\\Centuriton\\test_dataset.xlsx")
    #st.write(data)
    with row3_1:
        st.markdown("")
        see_data = st.expander('Click here to see the synthetic data  👉')
        with see_data:
            st.dataframe(data=data.reset_index(drop=True))
    data = pd.read_csv("D:\\Centuriton\\test_dataset 1.csv")
    l=['Actual Density Score','Predicted Density Score']
    data_480p = pd.DataFrame(data[data['Group'] == '480p'],columns=l)
    data_720p = pd.DataFrame(data[data['Group'] == '720p'], columns=l)
    data_1080p = pd.DataFrame(data[data['Group'] == '1080p'], columns=l)
    c1,c2,c3=st.columns(3)
    with c1:
        st.write('480p')
        st.line_chart(data_480p)
        st.markdown("Accuracy : **_47.9267%_**")
    with c2:
        st.write('720p')
        st.line_chart(data_720p)
        st.markdown("Accuracy : **_53.8617%_**")
    with c3:
        st.write('1080p')
        st.line_chart(data_1080p)
        st.markdown("Accuracy : **_70.5871%_**")
    st.subheader("Overall Accuracy: ")
    st.markdown(" **_55.6601%_**")


st.markdown("----", unsafe_allow_html=True)
