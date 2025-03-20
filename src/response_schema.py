yes_or_no_schema = {
    "title": "question",
    "description": "question",
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "boolean",
            "description": "If the answer implies 'yes', then final_answer is true; otherwise, final_answer is false"
        }
    },
    "required": ["final_answer"],
}


yes_or_no_schema_v2 = {
    "title": "question",
    "description": "Answer a Yes/No question with a clear explanation.",
    "type": "object",
    "properties": {
        "explain": {
            "type": "string",
            "description": (
                "Provide a logical and clear explanation. "
                "Ensure the explanation and final answer are fully aligned. "
            ),
        },
        "final_answer": {
            "type": "boolean",
            "description": (
                "The final yes/no answer based strictly on the explanation above. "
                "If the explanation implies 'Yes', return true. If it implies 'No', return false. "
                "Ensure final_answer matches explain exactly. "
            ),
            "examples": [True, False]
        }
    },
    "required": ["explain", "final_answer"],
    "examples": [
        {
            "explain": "No, the person in the image is a male.",
            "final_answer": False
        },
        {
            "explain": "Yes, the sky is blue on a clear day.",
            "final_answer": True
        },
        {
            "explain": "Không, người trong ảnh là nam.",
            "final_answer": False
        },
        {
            "explain": "Có, trong ảnh có trên 3 người.",
            "final_answer": True
        }
    ]
}

yes_or_no_schema_vannhk = {
    "title": "question",
    "description": "Answer a Yes/No question with a clear explanation.",
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "boolean",
            "description": (
                "The final yes/no answer strictly depends on the explanation below. "
                "If the explanation states 'Yes', final_answer must be True. "
                "If the explanation states 'No', final_answer must be False. "
                "Ensure that the final_answer is logically aligned with the explanation."
            ),
            "examples": [True, False]
        },
        "explain": {
            "type": "string",
            "description": (
                "Provide a logical and clear explanation."
                "Ensure the explanation and final answer are fully aligned. "
                # "Do not make assumptions beyond what is visible in the image."
            ),
        },
    },
    "required": ["final_answer", "explain"],
    "examples": [
        {
            "explain": "No, the person in the image is a male.",
            "final_answer": False
        },
        {
            "explain": "Yes, the person in the image is a female.",
            "final_answer": True
        },
        {
            "explain": "Yes, the sky is blue on a clear day.",
            "final_answer": True
        },
        {
            "explain": "Không, người trong ảnh là nam.",
            "final_answer": False
        },
        {
            "explain": "Có, trong ảnh có trên 3 người.",
            "final_answer": True
        }
    ]
}