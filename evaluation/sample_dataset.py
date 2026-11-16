
# A small dataset for learning RAG evaluation.
# These are illustrative examples, not live RAG outputs.

evaluation_data = [
    {
        "user_input": "How often must employees change passwords?",
        "retrieved_contexts": [
            "Employees must change their network passwords every 90 days."
        ],
        "response": "Employees must change passwords every 90 days.",
        "reference": "Employees must change their network passwords every 90 days."
    },
    {
        "user_input": "When is the IT helpdesk available?",
        "retrieved_contexts": [
            "The IT helpdesk operates from 9 AM to 6 PM on weekdays."
        ],
        "response": "The IT helpdesk is available 24 hours a day.",
        "reference": "The IT helpdesk operates from 9 AM to 6 PM on weekdays."
    }
]

if __name__ == "__main__":
    for item in evaluation_data:
        print("Question:", item["user_input"])
        print("Answer:", item["response"])
        print("Reference:", item["reference"])
        print()
