import os
from dotenv import load_dotenv

class Config:
    environment = os.getenv("FLASK_ENV", "development")
    if environment == "production":
        load_dotenv(".env.production")
    else:
        load_dotenv(".env.development")
    SQLALCHEMY_DATABASE_URI = (
        "mssql+pyodbc://userdbowner:$P4ssdbowner01#@srv-db-east-us003.database.windows.net/db_cfa_dev"
        "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=Yes"
    )
    print("SQLALCHEMY_DATABASE_URI",SQLALCHEMY_DATABASE_URI)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
