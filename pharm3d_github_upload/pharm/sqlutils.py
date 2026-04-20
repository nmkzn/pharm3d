import pymysql

def search_pend_jobs(col='pending', jobtype='model'):
    try:
        connection = pymysql.connect(
        host='127.0.0.1',  # 数据库主机名
        port=3306,               # 数据库端口号，默认为3306
        user='shiyu',             # 数据库用户名
        passwd='PkJZaO.9h65',         # 数据库密码
        autocommit=True,
        charset='utf8'           # 字符编码
        )
        cursor = connection.cursor()
        connection.select_db("pharm3d")
        sql = f"select * from flaskjobs where status like '{col}' AND jobtype like '{jobtype}';"
        cursor.execute(sql)
        results = cursor.fetchone()
        cursor.close()
        connection.close()
        return results
    except Exception as e:
        return e
    
def alter_pend_jobs(jobid, status):
    try:
        connection = pymysql.connect(
        host='127.0.0.1',  # 数据库主机名
        port=3306,               # 数据库端口号，默认为3306
        user='shiyu',             # 数据库用户名
        passwd='PkJZaO.9h65',         # 数据库密码
        autocommit=True,
        charset='utf8'           # 字符编码
        )
        cursor = connection.cursor()
        connection.select_db("pharm3d")
        sql = f"update flaskjobs set status='{status}' where jobid='{jobid}';"
        cursor.execute(sql)
        results = cursor.fetchone()
        cursor.close()
        connection.close()
        return results
    except Exception as e:
        return e