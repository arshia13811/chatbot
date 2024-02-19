# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from openai import OpenAI
import streamlit as st



def load_data(file_path):
    # Load your dataset
    # Generate BERT embeddings for your dataset
    # This should return your dataset and a NearestNeighbors model trained on your embeddings
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

data = load_data('FreeCoursePartList.txt')



# # Initialize the tokenizer and model
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np

# Initialize the tokenizer and model
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)


# Extract descriptions from your data
descriptions = [item['Description'] for item in data]


# Function to encode texts
def encode_texts_in_batches(texts, batch_size=10, num_epochs=3):
    batched_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="tf")
        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
        batch_embeddings = outputs.last_hidden_state[:,0,:].numpy()
        batched_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    embeddings = np.vstack(batched_embeddings)
    return embeddings
# use the the enocoding function to generate embeddings
embeddings = encode_texts_in_batches(descriptions, batch_size=10, num_epochs=3)  

print(f"Generated {len(embeddings)} embeddings.")
print(descriptions)


knn_model = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embeddings)


def find_closest_data_points(user_question):
    # Convert the question to a BERT embedding
    question_embedding = encode_texts_in_batches([user_question])  # Use your actual function here
    
    # Use the KNN model to find the closest points
    distances, indices = knn_model.kneighbors(question_embedding)
    
    # Fetch the closest data points
    closest_data_points = [data[i] for i in indices[0]]
    return closest_data_points


# Function to construct messages for the OpenAI API call
def construct_messages(closest_data_points, user_question):
    context = "The following are the most relevant courses based on your needs:\n"
    for point in closest_data_points:
        context += f"based on the following data, answer the question: - Title: {point['Title']}, Description: {point['Description']}\n"
    
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_question}
    ]
    
    return messages



# Initialize your OpenAI client

# Load environment variables from .env file
# load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key= "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")

def get_chat_response(user_question):
    closest_data_points = find_closest_data_points(user_question)
    messages = construct_messages(closest_data_points, user_question)
    
    # Make the API call to OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    # Return the response
    return completion.choices[0].message



def process_question(user_question):
    # Find the closest data points to the question
    closest_data_points = find_closest_data_points(user_question)
    
    # Construct messages for the OpenAI API call
    messages = construct_messages(closest_data_points, user_question)
    
    # Make the API call to OpenAI to get the response
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # Return the generated response
    return completion.choices[0].message




def main():
    # Add a title and a description
    st.title("Chatbot for Course Recommendations")
    st.write("Enter your question to get course recommendations:")

    # Replace the sample question with user input
    user_question = st.text_input("Question:")

    if st.button("Submit"):
        if user_question:
            # Process the question
            response = process_question(user_question)
            closest_points = find_closest_data_points(user_question)
            
            # Display the response
            st.write("The closest points are:", closest_points)
            st.write("The Davincin answer is:", response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
