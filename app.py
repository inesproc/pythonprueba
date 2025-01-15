from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from Modelos.Preasignacion import ejecutar_preasignacion
from Modelos.DistribucionStock import ejecutar_distribucion_stock
from Modelos.Asignacion import ejecutar_asignacion
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, verify_jwt_in_request, get_jwt, get_jwt_identity
from dotenv import load_dotenv
import os
from configDB import Config
import jwt as jwt_token
import logging

environment = os.getenv("FLASK_ENV", "development")

# Cargar el archivo .env adecuado
if environment == "production":
    load_dotenv(".env.production")
else:
    load_dotenv(".env.development")

app = Flask(__name__)
app.config.from_object(Config)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
jwt = JWTManager(app)

db = SQLAlchemy(app)
# Ruta para preasignación
@app.route('/api/preasignacion', methods=['GET'])
def exec_preasignacion():
    try:
        # Obtener el token del encabezado
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token no proporcionado o formato incorrecto"}), 401

        # Extraer el token
        token = auth_header.split(" ")[1]
        try:
            # Decodificar el token sin verificar la firma
            jwt_payload = jwt_token.decode(token, options={"verify_signature": False})
        except Exception as e:
            return jsonify({"error": f"Error al decodificar el token: {str(e)}"}), 400

        # Verificar si el parámetro 'parameter' está presente y es 'true'
        if jwt_payload.get('parameter') != 'true':
            return jsonify({"error": "No autorizado"}), 403

        # Obtener el id_batch de los parámetros de consulta
        id_batch = request.args.get('id_batch')
        if not id_batch:
            return jsonify({"error": "id_batch is required"}), 400

        # Ejecutar lógica de preasignación
        ejecutar_preasignacion(id_batch)

        return jsonify({"message": "Preasignación ejecutada correctamente."}), 200

    except Exception as e:
        # Captura cualquier excepción y devuelve un mensaje de error
        return jsonify({"error": f"Hubo un error durante la ejecución: {str(e)}"}), 500

# Ruta para distribución de stock
@app.route('/api/distribucionStock', methods=['GET'])
# @jwt_required()
def exec_distribucionStock():
    try:
        # Obtener el token del encabezado
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token no proporcionado o formato incorrecto"}), 401

        # Extraer el token
        token = auth_header.split(" ")[1]
        try:
            # Decodificar el token sin verificar la firma
            jwt_payload = jwt_token.decode(token, options={"verify_signature": False})
        except Exception as e:
            return jsonify({"error": f"Error al decodificar el token: {str(e)}"}), 400

        # Verificar si el parámetro 'parameter' está presente y es 'true'
        if jwt_payload.get('parameter') != 'true':
            return jsonify({"error": "No autorizado"}), 403

        id_batch = request.args.get('id_batch')  
        if not id_batch:
            return jsonify({"error": "id_batch is required"}), 400

        # Aquí debes ejecutar tu lógica para la distribución
        ejecutar_distribucion_stock(id_batch)

        return jsonify({"message": "Distribución Stock ejecutada correctamente."}), 200

    except Exception as e:
        # Captura cualquier excepción y devuelve un mensaje de error
        return jsonify({"error": f"Hubo un error durante la ejecución: {str(e)}"}), 500

# Ruta para asignación
@app.route('/api/asignacion', methods=['GET'])
# @jwt_required()
def exec_asignacion():
    try:
        # Obtener el token del encabezado
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token no proporcionado o formato incorrecto"}), 401

        # Extraer el token
        token = auth_header.split(" ")[1]
        try:
            # Decodificar el token sin verificar la firma
            jwt_payload = jwt_token.decode(token, options={"verify_signature": False})
        except Exception as e:
            return jsonify({"error": f"Error al decodificar el token: {str(e)}"}), 400

        # Verificar si el parámetro 'parameter' está presente y es 'true'
        if jwt_payload.get('parameter') != 'true':
            return jsonify({"error": "No autorizado"}), 403
        
        ejecutar_asignacion()
        return jsonify({"message": "Asignación ejecutada correctamente."}), 200

    except Exception as e:
        # Captura cualquier excepción y devuelve un mensaje de error
        return jsonify({"error": f"Hubo un error durante la ejecución de asignación: {str(e)}"}), 500

if __name__ == '__main__':
    # Configura el host y el puerto aquí
    app.run(debug=True, host="0.0.0.0", port=5000)
