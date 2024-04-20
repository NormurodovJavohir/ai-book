 import re

class SimpleChatbot:
    def __init__(self):
        self.responses = {
            r'hello|hi|hey': ['Hello!', 'Hi there!', 'Hey!'],
            r'how are you\??': ['I am doing well, thank you!', 'Great, thanks for asking!', 'I am fine, how about you?'],
            r'(.*) name\??': ["My name is Chatbot. What's yours?", "I'm Chatbot. What can I do for you?", "I go by Chatbot. What about you?"],
            r'(.*) (age|old)\??': ["I don't have an age. I'm just a computer program.", "I'm ageless!", "Age is just a number, isn't it?"],
            r'(.*) (love|like) (.*)': ["I'm glad you love/like {}. It's great to hear that!", "That's wonderful! {} is amazing.", "{} sounds interesting."],
            r'(.*) (weather|forecast)\??': ["I'm sorry, I cannot provide weather information at the moment.", "You might want to check a weather website for that information."],
            r'(.*) (thank you|thanks)\??': ["You're welcome!", "No problem!", "Anytime!"],
            r'(.*) (sorry|apologize)\??': ["No need to apologize.", "It's okay.", "Don't worry about it."],
            r'bye|goodbye': ['Goodbye!', 'See you later!', 'Bye! Take care.']
        }

    def respond(self, user_input):
        for pattern, responses in self.responses.items():
            match = re.match(pattern, user_input.lower())
            if match:
                response = responses[0] 
                return response.format(*match.groups()) if '{}' in response else response

        return "I'm sorry, I didn't understand that."

# Example usage:
if __name__ == "__main__":
    chatbot = SimpleChatbot()
    
    print("Chatbot: Hi! I'm Chatbot. How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot.respond(user_input)
        print("Chatbot:", response)
