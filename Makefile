streamlit:
	streamlit run streamlit_app.py --server.port=4444

api:
	uvicorn api:app --reload
