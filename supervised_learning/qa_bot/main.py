
question_answer = __import__('0-qa').question_answer


with open('data/ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))