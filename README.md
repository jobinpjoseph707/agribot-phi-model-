# Agribot PHI Model

Agribot PHI Model is a Flask-based web application that integrates chat-based interactions with real-time weather updates and crop recommendations. The system uses a Retrieval Augmented Generation (RAG) pipeline and a language model to provide friendly responses as well as location-specific crop suggestions based on weather data.

## Overview

The application processes user queries in two main ways:

- **Plain Chat (No Location Detected):**  
  If the query (e.g., "hi") does not mention any of the predefined Indian states, the system constructs a prompt for a friendly response. The model is instructed to reply in a friendly tone, using a maximum of 5 short sentences.

- **Location-Based Query:**  
  If the query includes a location (e.g., "What is best crop for Punjab weather?"), the system:
  1. **Extracts the Location:**  
     The `extract_location(query)` method scans the query for any Indian state (e.g., Punjab, Kerala).
  2. **Fetches Weather Data:**  
     The `update_weather_data(location)` function calls `get_weather_data(location)` to retrieve current weather data and updates a local JSON cache.
  3. **Retrieval and Crop Suggestion:**  
     The RAG system is invoked via `rag.retrieve(query, weather)` to obtain additional context. Then, a detailed prompt is created:
     > "Pick one crop from RAG and suggest it for [location]'s weather in exactly 5 short sentences."
     
     This prompt is sent to the model using `ModelInference.infer(prompt)` to generate a crop suggestion.
  4. **Response Assembly:**  
     The final response combines the weather data, the RAG output, and the model’s crop suggestion.

## Code Flow Diagram

Below is a simplified flow diagram illustrating how a user query is processed:

```mermaid
flowchart TD
    A[User Input]
    B[Flask App (app.py)]
    C[BotInterface.process_query(query)]
    D{Is a Location Detected?}
    E[Extract Location from Query]
    F[Fetch & Update Weather Data]
    G[Call RAGSystem.retrieve(query, weather)]
    H[Construct prompt: "Pick one crop..." ]
    I[Call ModelInference.infer(prompt)]
    J[Return Combined Response (Weather, RAG, Crop Suggestion)]
    K[Construct friendly chat prompt]
    L[Call ModelInference.infer(prompt)]
    M[Return Friendly Chat Response]

    A --> B
    B --> C
    C --> D
    D -- Yes --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    D -- No --> K
    K --> L
    L --> M
