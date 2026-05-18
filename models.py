from sqlalchemy import Column, Integer, Text, TIMESTAMP, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from database import Base


from sqlalchemy import Column, Integer, Text, TIMESTAMP, LargeBinary, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(Text)
    file_path = Column(Text)
    image_data = Column(LargeBinary, nullable=True)
    upload_time = Column(TIMESTAMP, server_default=func.now())
    client_ip = Column(Text)

    ocr_firma_adi = Column(Text)
    ocr_sozlesme_no = Column(Text)
    ocr_tutar = Column(Text)
    ocr_fatura_turu = Column(Text)
    ocr_raw_json = Column(JSONB)

    vlm_firma_adi = Column(Text)
    vlm_sozlesme_no = Column(Text)
    vlm_tutar = Column(Text)
    vlm_fatura_turu = Column(Text)
    vlm_raw_json = Column(JSONB)

    # OCR süre takibi
    ocr_started_at = Column(TIMESTAMP, nullable=True)
    ocr_finished_at = Column(TIMESTAMP, nullable=True)
    ocr_duration_seconds = Column(Float, nullable=True)