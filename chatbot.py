import cohere

# Initialize Cohere client with your API key
cohere_api_key = "aFMeD1nntPUkFhSj5VYd5f5Ghwe7zVWguZ88ymtw"
co = cohere.ClientV2(api_key=cohere_api_key)  # Using ClientV2 for chat streaming

# Chatbot response function using Cohere API (chat_stream)
def get_chatbot_response(user_input, temperature=0.5):
    # Define casual greetings
    greetings = ["hello", "hi", "hey", "howdy", "yo", "greetings"]
    
    # Check if the input is a greeting
    if any(greeting in user_input.lower() for greeting in greetings):
        return "Hello! How can I assist you with your astrology questions today?"
    
    # Handle the general question about astrology
    if 'what is astrology' in user_input.lower():
        return ("Astrology is the study of the positions and movements of celestial bodies, like the planets, sun, and moon, "
                "and their potential influence on events and human behavior. It suggests that the alignment of these bodies "
                "at the time of your birth can provide insights into your personality, relationships, and even future events. "
                "The zodiac signs, like Aries, Taurus, Gemini, etc., are part of this belief system, each representing different "
                "characteristics and traits.")
    
    # Handle astrology-related queries generally
    if 'astrology' in user_input.lower() or 'horoscope' in user_input.lower():
        prompt = f"You are an astrology expert. Answer the following astrology-related question: {user_input}"

    # Handle questions about specific signs, like Capricorn, Leo, or Aries
    zodiac_signs = ["leo", "aries", "capricorn", "taurus", "virgo", "libra", "scorpio", "sagittarius", 
                    "aquarius", "pisces", "gemini", "cancer"]
    
    if any(sign in user_input.lower() for sign in zodiac_signs):
        sign = next(sign for sign in zodiac_signs if sign in user_input.lower())
        prompt = f"Explain the astrology of {sign.capitalize()}. What are the personality traits, ruling planet, and key astrological influences of this zodiac sign?"

    else:
        # Default general astrology prompt if the input doesn't match specific criteria
        prompt = f"You are an astrology bot. Answer questions about horoscopes, astrology, and the zodiac signs. Question: {user_input}"

    # Start streaming the response
    model = "command-r-plus-08-2024"  # Make sure you have access to this model for streaming
    messages = [{"role": "user", "content": user_input}]
    
    try:
        # Use the chat_stream method
        response = co.chat_stream(
            model=model,
            messages=messages
        )

        # Collect and display the streaming response
        response_text = ""
        for event in response:
            if event.type == "content-delta":
                response_text += event.delta.message.content.text

        return response_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
user_input = "What is the astrology of Gemini?"
response = get_chatbot_response(user_input)
print(response)
