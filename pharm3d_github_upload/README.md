# pharm3d website application

## Introduction
This repo is design to deploy pharm3d tool on server.
整个服务一共分为两个部分，两个部分通过数据表（flaskjobs）相互连接：
    1. 网站服务部分：用户注册（包括邮箱验证）、登录、查看文档、提交任务、获取结果等内容。
    2. 计算服务部分：embed_script.py和screen_script.py两个脚本，会每20分钟自动运行一次，查找数据表（flaskjobs）中处于pending状态的任务，将其状态修改为computing，并且运行该任务。对于运行成功的任务，脚本会将其状态修改为success，运行失败的任务的状态会被修改为failed。

## mysql配置
    1. 在服务器上安装mysql服务器，并且开启mysql数据库
    2. 创建用户, 并且指定用户密码
    3. 在mysql软件中创建数据库：pharm3d
    命令为：create database pharm3d default charset utf8 collate utf8_general_ci;
    4. 为用户分配数据库权限
 
    5. 在pharm3d数据库中创建数据表（flaskusers）用于存储用户登录的数据
```sql
create table flaskusers
(
id int unsigned not null auto_increment primary key,
username varchar(50) NOT NULL,
password varchar(60) NOT NULL,
email varchar(50) NOT NULL,
is_active BOOLEAN DEFAULT TRUE,
is_admin BOOLEAN DEFAULT FALSE
);
```
    该数据表一共有6列，分别为：
    +-----------+--------------+------+-----+---------+----------------+
    | Field     | Type         | Null | Key | Default | Extra          |
    +-----------+--------------+------+-----+---------+----------------+
    | id        | int unsigned | NO   | PRI | NULL    | auto_increment |
    | username  | varchar(50)  | NO   |     | NULL    |                |
    | password  | varchar(60)  | NO   |     | NULL    |                |
    | email     | varchar(50)  | NO   |     | NULL    |                |
    | is_active | tinyint(1)   | YES  |     | 1       |                |
    | is_admin  | tinyint(1)   | YES  |     | 0       |                |
    +-----------+--------------+------+-----+---------+----------------+
    6. 在pharm3d数据库中创建数据表（flaskmessages）用于存储用户留言的数据
```sql
create table flaskmessages
(
id int unsigned not null auto_increment primary key,
email varchar(50) NOT NULL,
phone varchar(50) NOT NULL,
message varchar(150) NOT NULL,
name varchar(20) NOT NULL
);
```
    该数据表有5列，分别为：
    +---------+--------------+------+-----+---------+----------------+
    | Field   | Type         | Null | Key | Default | Extra          |
    +---------+--------------+------+-----+---------+----------------+
    | id      | int          | NO   | PRI | NULL    | auto_increment |
    | email   | varchar(50)  | NO   |     | NULL    |                |
    | phone   | varchar(50)  | NO   |     | NULL    |                |
    | message | varchar(150) | NO   |     | NULL    |                |
    | name    | varchar(20)  | NO   |     | NULL    |                |
    +---------+--------------+------+-----+---------+----------------+
    7. 在pharm3d数据库中创建数据表（flaskjobs）用于存储用户提交的任务
```sql
create table flaskjobs 
(
    id int unsigned not null auto_increment primary key,
    email varchar(50) NOT NULL,
    jobid varchar(50) NOT NULL,
    jobtype varchar(30) NOT NULL,
    status varchar(30) NOT NULL DEFAULT 'pending',
    parameters json NOT NULL
);
```
    该数据表有6列，分别为：
    +------------+--------------+------+-----+---------+----------------+
    | Field      | Type         | Null | Key | Default | Extra          |
    +------------+--------------+------+-----+---------+----------------+
    | id         | int unsigned | NO   | PRI | NULL    | auto_increment |
    | email      | varchar(50)  | NO   |     | NULL    |                |
    | jobid      | varchar(50)  | NO   |     | NULL    |                |
    | jobtype    | varchar(30)  | NO   |     | NULL    |                |
    | status     | varchar(30)  | NO   |     | pending |                |
    | parameters | json         | NO   |     | NULL    |                |
    +------------+--------------+------+-----+---------+----------------+
    
    8. 在pharm3d数据库中创建数据表（flaskcodes）用于存储向用户发送的验证码
```sql
create table flaskcodes
(
    id int unsigned not null auto_increment primary key,
    email varchar(50) NOT NULL,
    code varchar(30) NOT NULL
);
```
    该数据表有3列，分别为：
    +-------+--------------+------+-----+---------+----------------+
    | Field | Type         | Null | Key | Default | Extra          |
    +-------+--------------+------+-----+---------+----------------+
    | id    | int unsigned | NO   | PRI | NULL    | auto_increment |
    | email | varchar(50)  | NO   |     | NULL    |                |
    | code  | varchar(30)  | NO   |     | NULL    |                |
    +-------+--------------+------+-----+---------+----------------+

## PYTHON环境
    flask==2.1.3
    pandas==2.0.3
    python==3.8.18
    pymysql==1.0.2
    flask_login==0.6.3
    flask_mail==0.10.0

## 服务器启动方法
    请确保5000端口是开放的
    python app.py
    服务启动之后，直接在浏览器中访问：http://127.0.0.1:5000/

## Pharm3D grid启动方法 
    这一个部分的python可以与服务器的python不是同一个python环境
    python embed_script.py #开启modeling部分的任务
    python screen_script.py #开启screening部分的任务
