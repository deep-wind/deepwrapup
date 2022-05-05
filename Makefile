streamlit:
	streamlit run streamlit_app.py --server.port=88

api:
	uvicorn api:app --reload
