from minio import Minio
from minio.error import S3Error

# Khoi tao client
client = Minio(
    "localhost:9000",
    access_key="vannhk", # Minio user
    secret_key="123456789", # Minio Password
    secure=False,
)

# Tao bucket (neu chua ton tai)
bucket_name = "rag-chatbot-data"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

file_path = "/home/vbd-vanhk-l1-ubuntu/PycharmProjects/PythonProject/data/VBD_IVA_HDSD.pdf"
object_name = "pdfs/vbd_iva_hdsd.pdf"

client.fput_object(bucket_name, object_name, file_path)
print(f"[UPLOAD] Đã upload {object_name}")
