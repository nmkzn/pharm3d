from flask import Flask, render_template, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
import pandas as pd
import json, time, os
from webutils import hashmd5
from webutils import write_message, write_users, search_users, search_users_id
from webutils import search_jobs_email, write_jobs, v_code
from webutils import search_code_email, write_code_email, update_code_email
from gevent import pywsgi

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

app.config['MAIL_SERVER']='smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = '2572827223'
app.config['MAIL_PASSWORD'] = 'gpalcfjryxcedhje'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
flaskMail = Mail(app)

class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route("/")
def hello():
    return render_template('home.html')

@app.route("/home", methods=['GET'])
def about():
    return render_template('home.html')

@app.route("/tutorial", methods=['GET'])
def tutorial():
    return render_template('tutorial.html')

@app.route("/model", methods=['GET'])
@login_required
def model():
    # generate jobid
    user_id = current_user.id
    jobid = str(user_id).rjust(4,'0')+str(int(time.time()))
    return render_template('model_submit.html', jobid=jobid)

@app.route("/model", methods=['POST'])
@login_required
def model_submit():
    # generate jobid
    user_id = current_user.id
    email = search_users_id(user_id)[3]
    parameters={}
    parameters["conformer"]=str(request.form['conformer-train'])
    jobid = request.form['jobPath']
    result = write_jobs(jobid, email, 'model', json.dumps(parameters))
    return render_template('model_submit.html', message=result)

@app.route("/screen", methods=['GET'])
@login_required
def screen():
    # generate jobid
    user_id = current_user.id
    jobid = str(user_id).rjust(4,'0')+str(int(time.time()))
    return render_template('screen_submit.html', jobid=jobid)

@app.route("/screen", methods=['POST'])
@login_required
def screen_submit():
    # generate jobid
    user_id = current_user.id
    email = search_users_id(user_id)[3]
    jobid = request.form['jobPath']
    parameters={}
    parameters["conformer"]=str(request.form['conformer-screen'])
    parameters["nfeat"]=str(request.form['pharmacophores'])
    result = write_jobs(jobid, email, 'screen', json.dumps(parameters))
    return render_template('screen_submit.html', message=result)

@app.route("/uploadfile", methods=['POST'])
@login_required
def uploadfile():
    try:
        jobid = request.form['jobPath']
        jobpath = f"static/jobfolder/{jobid}/"
        filename = request.form['filename']
        if not os.path.exists(jobpath):
            os.makedirs(jobpath)
        myUploadFile = request.files['myUploadFile']
        myUploadFile.save(jobpath+filename)
        return_data={'code':0,'msg':'Upload success','path':myUploadFile.filename}
        return jsonify(return_data)
    except:
        return_data={'code':-1,'msg':'Upload failed'}
        return jsonify(return_data)

@app.route("/view", methods=['GET'])
@login_required
def view():
    jobid=request.args["jobid"]
    nfeat=request.args["nfeat"]
    fold=request.args["fold"]
    molpath = f"static/jobfolder/{jobid}/crystal_ligand.mol2"
    featpath = f"static/jobfolder/{jobid}/feat{nfeat}_mymodel_{fold}fold.pth.pdb"
    if nfeat==str(20):
        m_radius = 1.2
    else:
        m_radius = 0.3
    return render_template('view_embed.html', molpath=molpath, featpath=featpath, m_radius=m_radius)

@app.route("/jobs", methods=['GET'])
@login_required
def jobs():
    user_id = current_user.id
    email = search_users_id(user_id)[3]
    if "@" not in email:
        return render_template('home.html')
    else:
        results = search_jobs_email(email)
        df = pd.DataFrame(results)
        if len(results) > 0:
            df.columns = ['id','email','jobid','jobtype','status','parameters']
            views=[]
            for idx in df.index:
                s = df['status'].to_list()[idx]
                jobid = df['jobid'].to_list()[idx]
                jobtype = df['jobtype'].to_list()[idx]
                if s == "success" and jobtype=="model":
                    views.append(f'''<a href='result_embed?jobid={jobid}'>view</a>''')
                elif s == "success" and jobtype=="screen":
                    views.append(f'''<a href='static/jobfolder/{jobid}/matched_molIdx.sdf' download>Download sdf</a><br><a href='static/jobfolder/{jobid}/matched_molIdx.txt' download>Download index</a>''')
                elif s == "failed":
                    views.append(f'''<a href='static/jobfolder/{jobid}/job.log' download>Log File</a>''')
                else:
                    views.append('view')
            df["results"] = views
        return render_template('jobs.html', df=df.to_html(index=None, justify="center", escape=False, border=2, classes=['dataframe', 'table', 'table-striped']))

@app.route("/contacts", methods=['GET'])
def contacts():
    return render_template('contacts.html')

@app.route("/sendcode", methods=['GET'])
def sendcode():
    try:
        mail=request.args["mail"]
        result = search_code_email(mail)
        r_code = v_code()
        if result==None:
            write_code_email(mail, r_code)
        else:
            update_code_email(mail, r_code)
        msg = Message('Your 6-digit verification code from Pharm3D Grid website.', sender = ("Pharm3dGrid",'2572827223@qq.com'), recipients = [mail])
        msg.body = "Hello, New user. Welcome to our server. Your 6-digit verification code is " + r_code
        flaskMail.send(msg)
        return "Code send successfully"
    except Exception as e:
        print(e)
        return "Code send failed"

@app.route("/result_embed", methods=['GET'])
@login_required
def result_embed():
    dataframe=pd.DataFrame()
    jobid=request.args["jobid"]
    dirname = f"static/jobfolder/{jobid}/"
    accuracy=pd.read_csv(os.path.join(dirname,'accuracy'))
    dataframe["fold"] = accuracy["fold"]
    dataframe["accuracy"] = accuracy["accuracy"]
    dataframe["MSE_Loss"] = ["<a href='"+ dirname + "Fold" + str(fold) + "_loss.png" +"'>view</a>" for fold in accuracy["fold"]]
    dataframe["Box Boundary"] = ["<a href='"+ dirname + "box_info.txt" +"' download>Download</a>" for _fold in accuracy["fold"]]
    dataframe["Attention weight"] = ["<a href='"+ dirname + "indicesAttWtPd_" + str(fold) + "fold.csv" +"' download>Download</a>" for fold in accuracy["fold"]]
    dataframe["Pharmacophores"] = [f"<a href='view?jobid={jobid}&fold={fold}&nfeat=20' target='_blank'>view(20 features)</a><br><a href='view?jobid={jobid}&fold={fold}&nfeat=100' target='_blank'>view(100 features)</a>" for fold in accuracy["fold"]]

    nfeat=20
    fold=1
    molpath = f"static/jobfolder/{jobid}/crystal_ligand.mol2"
    featpath = f"static/jobfolder/{jobid}/feat{nfeat}_mymodel_{fold}fold.pth.pdb"
    return render_template('result_embed.html', df=dataframe.to_html(index=None, justify="center", escape=False, border=2, classes=['dataframe', 'table', 'table-striped']), jobid=jobid, molpath=molpath, featpath=featpath)

@app.route("/result_screen", methods=['GET'])
@login_required
def result_screen():
    return render_template('result_screen.html')

@app.route("/login", methods=['GET'])
def login():
    message_tmp="Login is required before you use our website."
    return render_template('login.html', message=message_tmp)

@app.route("/login", methods=['POST'])
def login_post():
    email = request.form['email']
    password = hashmd5(request.form['password'])
    result = search_users(email)
    message_tmp = "Wrong Password or Email address."
    try:
        search_password = result[2]
        if password == search_password:
            user_id = result[0]
            user = load_user(user_id)
            login_user(user)
            return render_template('home.html')
        if password != search_password:
            return render_template('login.html', message=message_tmp)
    except:
        return render_template('login.html', message=message_tmp)
  

@app.route("/register", methods=["GET"])
def register():
    result=""
    return render_template('register.html', result=result)

@app.route("/register", methods=['POST'])
def register_post():
    name = request.form['name']
    email = request.form['email']
    code = request.form['code']
    password = hashmd5(request.form['password'])
    result = search_users(email)
    if result != None:
        return render_template('register.html', result="This email has already been used!")
    mail_code_result=search_code_email(email)
    if mail_code_result == None:
        return render_template('register.html', result="The verification code has expired!")
    else:
        mysql_code = mail_code_result[2]
        if mysql_code != code:
            return render_template('register.html', result="The verification code seems not right!")
    result = write_users(name, email, password)
    return render_template('register.html', result=result)

@app.route("/contacts", methods=['POST'])
def contacts_post():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    message = request.form['message']
    result = write_message(name, email, phone, message)
    msg = Message('New contact info from pharm3dgrid.', sender = ('Pharm3dGrid Contacts','2572827223@qq.com'), recipients = ["contact@cadd2drug.org"])
    tem_message = "name: " + name + "\n"
    tem_message += "email: "+email + "\n"
    tem_message += "phone: "+phone + "\n"
    tem_message += "message: "+ message
    msg.body = tem_message
    flaskMail.send(msg)
    return render_template('contacts.html', message=result)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    message = 'Logged out successfully!'
    return render_template('home.html', message=message)

if __name__ == "__main__":
    # app.run(host = '0.0.0.0' ,port = 5000, debug = 'True')
    server = pywsgi.WSGIServer(('0.0.0.0',5000),app)
    server.serve_forever()
