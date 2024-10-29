from flask import Flask, request, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import io
import base64

app = Flask(__name__)

# Perform clustering and return the results as CSV and plot in base64
def perform_clustering(data, n_clusters):
    try:
        # Keep only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.empty:
            raise ValueError("No numeric data available for clustering.")
        
        # Handle missing values by imputing the mean
        imputer = SimpleImputer(strategy='mean')
        numeric_data_imputed = imputer.fit_transform(numeric_data)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300)
        cluster_labels = kmeans.fit_predict(numeric_data_imputed)
        data['Cluster'] = cluster_labels

        # Convert clustered data to CSV format (as string)
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # Plot the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=numeric_data_imputed[:, 0], y=numeric_data_imputed[:, 1], hue=cluster_labels, palette='viridis')
        plt.title(f'Clustering with {n_clusters} Clusters')

        # Save plot to buffer and encode it to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()

        return csv_string, img_base64
    except Exception as e:
        print(f"Error during clustering: {e}")
        return None, None

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        # Get the uploaded file from the request
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Read the file based on extension
        filename = file.filename
        if filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        elif filename.endswith('.json'):
            data = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Get clustering parameters from the request
        n_clusters = int(request.form.get('n_clusters', 5))

        # Perform clustering
        csv_string, img_base64 = perform_clustering(data, n_clusters)
        if csv_string is None or img_base64 is None:
            return jsonify({"error": "Error during clustering"}), 500

        # Return the clustered data and plot as a JSON response
        return jsonify({
            "clustered_data": csv_string,
            "plot_base64": img_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)