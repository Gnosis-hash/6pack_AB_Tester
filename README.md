#  6-Pack AB Test App

A web-based application designed to integrate seamlessly with your BigQuery project, enabling you to execute data queries and conduct A/B testing between two product events.

If you choose to fork this repository, please note that you will need to generate a custom secret file containing your service account JSON key. For detailed instructions on creating a service key and adding it to your Streamlit account's secret file, refer to the official Streamlit documentation provided in the link below:

https://docs.streamlit.io/develop/tutorials/databases/bigquery

When launching the app below, you can use the following query to get started that will utilize my GBQ project:

   ```
   SELECT * FROM `ab-test-app-452422.testdata.6pack_data`
   ```

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://6packabtest.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
