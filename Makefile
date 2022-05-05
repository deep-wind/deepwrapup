streamlit:
	streamlit run streamlit_app.py --server.port=8501

api:
	uvicorn api:app --reload
