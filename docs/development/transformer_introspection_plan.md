# Plan: Implement Transformer Introspection

## Goal

The objective is to add introspection capabilities to the transformer architecture. This will expose internal states, such as attention weights and layer outputs, for analysis, debugging, and a deeper understanding of the model's decision-making process.

This will be accomplished by:

1. Defining data structures to capture the internal states during a forward pass.
2. Modifying the forward pass to populate these structures.
3. Exposing the captured data through a new API endpoint.
4. Creating a client-side tool to visualize the data.

## Step-by-Step Implementation Plan

### 1. Define Introspection Data Structures

A set of dedicated data structures will be created to hold the introspection data in an organized manner.

* **Action**: Create a new header file `include/Introspection.hpp`.
* **Contents**:
  * `AttentionIntrospection`: A struct to store attention weight matrices from `MultiHeadAttention` and `CrossAttention` layers.
  * `EncoderLayerIntrospection`: A struct to store the outputs of sub-layers within an encoder block (attention and feed-forward) and the `AttentionIntrospection` data.
  * `DecoderLayerIntrospection`: A struct similar to the encoder's, but including data for both self-attention and cross-attention.
  * `TransformerIntrospection`: A top-level struct containing vectors of `EncoderLayerIntrospection` and `DecoderLayerIntrospection` objects, providing a complete view of the entire transformer stack.

### 2. Modify the Forward Pass to Populate Introspection Data

The core transformer components will be updated to populate the new data structures during computation.

* **Action**: Update the `forward` methods in the following classes:
  * `EncoderDecoderModel`
  * `LLMEncoder` & `LLMDecoder`
  * `EncoderBlock` & `DecoderBlock`
  * `MultiHeadAttention` & `CrossAttention`
* **Details**:
  * Each `forward` method will be overloaded or updated to accept an optional pointer to a `TransformerIntrospection` object.
  * When this pointer is provided, the method will fill in the relevant part of the introspection object with the computed internal states (e.g., `MultiHeadAttention` will save its `cached_attention_weights`).
  * This will be done efficiently, reusing existing cached values where possible.

### 3. Expose Introspection Data via a New API Endpoint

A new API endpoint will be created to provide access to the introspection data.

* **Action**: Modify `src/api.cpp` to add a new `/chat_introspect` endpoint.
* **Functionality**:
  * This endpoint will accept the same input as the existing `/chat` endpoint.
  * It will call a new `Chatbot::chat_with_introspection` method.
  * This method will execute the forward pass while populating the `TransformerIntrospection` object.
  * The resulting introspection object will be serialized to JSON and returned to the client.

### 4. Develop a Client-Side Visualization Tool

To make the introspection data useful, a tool will be developed to visualize it.

* **Action**: Create a Python script `scripts/visualize_attention.py`.
* **Features**:
  * The script will call the `/chat_introspect` endpoint with sample input.
  * It will parse the JSON response.
  * Using libraries like `matplotlib` and `seaborn`, it will generate and save heatmap images of the attention weights for each layer and head.

## Verification Plan

* **API Endpoint**: Test the `/chat_introspect` endpoint with `curl` or a similar tool to ensure it returns a well-formed JSON payload with the expected structure.
* **Data Correctness**: During a debugging session, set breakpoints and compare the live values of attention weights and layer outputs with the data returned by the API to ensure accuracy.
* **Visualization**: Run the `visualize_attention.py` script and inspect the generated heatmaps. The attention patterns should be plausible (e.g., showing expected relationships between tokens) and the tool should run without errors.
