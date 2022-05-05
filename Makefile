streamlit:
	streamlit run streamlit_app.py --server.port=80

api:
	uvicorn api:app --reload
