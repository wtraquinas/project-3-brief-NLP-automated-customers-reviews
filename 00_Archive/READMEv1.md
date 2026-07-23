


# Project | Business Case: Automated Customer Reviews

<br>

## Project Goal

This project aims to develop a product review system powered by NLP models that aggregate customer feedback from different sources. The key tasks include classifying reviews, clustering product categories, and using generative AI to summarize reviews into recommendation articles.





<br>



---


---

## MODEL TRACKING SPREADSHEET

| Model                                        | Feature extraction   | Training model          | Accuracy | test    | Notes                                                                                                                |
|----------------------------------------------|----------------------|-------------------------|----------|---------|----------------------------------------------------------------------------------------------------------------------|
| 1. Sentiment Analysis                                   |                | distilbert-base-uncased        | %   |         |                                                                                             |
| 2. Product Category Clustering                                   |    | KMeans + TF-IDF + ngrams        | %      |         |                                         |
| 3. Product Review Summarization                                   | | meta-llama/Llama-3.2-3B-Instruct          | %      |         |                                   |
| 4.                                  |                  |  |      |         |                                                                                                  |
| 5.                               |      |  |       |         |                                                                                 |
| 6.                                  |  |          | %   |         |                                                                                                                      |

<br>

---



## Deployment

### Application hosted on Vercel

<center>
<table>
  <td><img src="/images/deploy_streamlit_02_pred_260710.jpg" height="420" alt="Horse prediction"></td>
  <td> &nbsp &nbsp </td>
  <td><img src="/images/deploy_streamlit_02_pred_260710.jpg" height="420" alt="Dookie prediction"></td>
</table>
</center>



---

## Live Demo

🌐 **Vercel Hosting API and Application**

:white_check_mark: Application Frontend: 
https://p3nlp-customers-reviews.vercel.app/


:white_check_mark: API Health endpoint: 
https://p3nlp-customers-reviews.vercel.app/api/health

:white_check_mark: API Prediction endpoint: 
https://p3nlp-customers-reviews.vercel.app/api/predict


<br>


---

## Problem Statement

With thousands of reviews available across multiple platforms, manually analyzing them is inefficient. This project seeks to automate the process using NLP models to extract insights and provide users with valuable product recommendations.

<br>

---

## Datasets

**Primary Dataset - Consumer Reviews of Amazon Products** :
- https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data
- This is a list of over 34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and more provided by Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product.

     - File: `original_kaggle_data_download_archive.zip`
     - containing 3 csv files, with data to be to be merged



<br>

---

## Main Tasks


### 1. Build a model for Sentiment Analysis



<br>

---

### 2. Categorization with K-Means Clustering

**7 Clusters --> 7 Categories:**
     1. Tablets
     2. Tablets with Alexa
     3. Tablets for Kids
     4. E-Readers
     5. Smart Assistants
     6. Chargers & Adapters
     7. Others



<br>

---

### 3. Summarization




---

--- 


## Pipeline

```

```

### Text Preprocessing



### Feature Engineering



---

# Results


```

---

# Setup & Installation


```

---

# Project Structure

```

```

---

# Tech Stack



---

# Future Improvements



---

# Authors

**Antonio Traquinas** - https://github.com/wtraquinas/

and 

**David Tayebwa** - https://github.com/kerondavid-debug/

AI Engineering | Machine Learning | NLP

---

## Acknowledgements

Special Thanks to Luis Junco and Tejal Bhatti, for the inestimable assistance.

This project was developed as part of the **IronHack AI Engineering Bootcamp**, demonstrating the complete machine learning workflow from data preprocessing to deployment of an interactive web application.

