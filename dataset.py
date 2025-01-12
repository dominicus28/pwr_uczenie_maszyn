import pandas as pd
import numpy as np
# import api_calls

def prepare_dataset():
    df = pd.read_csv("spam.csv", encoding="latin1", usecols=['v1', 'v2'])

    X = np.array(df['v2'].tolist())
    y = np.array([1 if value == "spam" else 0 for value in df['v1'].tolist()])

    return X, y

def generate_synthetic_spam(X, y):
    print(X)
    print(X.shape)
    spam_texts = X[y == 1]
    valid_texts = X[y == 0]
    n_samples_to_generate = len(valid_texts) - len(spam_texts)
    print(f"Raw dataset length: {len(X)}")
    print(f"Valid messages: {len(valid_texts)}")
    print(f"Spam messages: {len(spam_texts)}")
    print(f"Synthetic spam messages which will be generated: {n_samples_to_generate}")

    # with open('raw_synthetic_spam.txt', 'a+') as f:
    #     for _ in range(n_samples_to_generate):
    #         generated_spam = api_calls.generate_response()
    #         f.write(f"{str(generated_spam)}\n")
    #     f.close()

# X, y = prepare_dataset()
# generate_synthetic_spam(X, y)

    # Example dataset
    # X = [
    #     "Win a free iPhone now! Click here to claim your prize!",
    #     "Hello, how are you doing today?",
    #     "Congratulations, you've won a $1000 gift card!",
    #     "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    #     "URGENT: Your account has been compromised. Please reset your password.",
    #     "Hey, are we still on for lunch this Friday?",
    #     "Claim your free vacation now by clicking this link!",
    #     "Meeting rescheduled to next week. Let me know your availability.",
    #     "Exclusive offer! Buy one, get one free!",
    #     "Your order has been shipped and is on its way.",
    #     "Win big! Participate in our survey and stand a chance to win $500!",
    #     "Looking forward to our meeting next Monday at 2 PM.",
    #     "Don’t miss out on this limited-time discount code: SAVE20!",
    #     "Hey, just checking in to see how you are doing.",
    #     "Congratulations on your new job! Let’s celebrate soon.",
    #     "Get a free trial of our premium subscription today!",
    #     "Can you send me the details for the project report?",
    #     "Your subscription has been renewed successfully.",
    #     "URGENT: Suspicious login detected on your account. Verify now.",
    #     "Reminder: Submit your assignment by 5 PM today.",
    #     "Final chance! Click here to redeem your reward.",
    #     "Good afternoon, I hope you’re doing well.",
    #     "You’ve been selected to receive a $200 gift voucher!",
    #     "Please confirm your delivery address for the shipment.",
    #     "Your account statement is now available online.",
    #     "Breaking news: Major updates in your local area!",
    #     "Earn cashback on every purchase with our new card.",
    #     "Can you review the attached document before our call?",
    #     "Don’t forget to attend the webinar tomorrow at 3 PM.",
    #     "Check out the latest deals in our online store!",
    #     "Hello, just reminding you about the dinner this weekend.",
    #     "Your parcel is out for delivery and will arrive soon.",
    #     "Hot sale! Discounts up to 70% on selected items!",
    #     "Click here to secure your entry to the contest!",
    #     "Meeting minutes from today’s session are attached.",
    #     "Join us for a free workshop on personal finance!",
    #     "Verify your email to complete the registration process.",
    #     "Reminder: Your car service is due next Monday.",
    #     "Congratulations! You’ve been shortlisted for the interview.",
    #     "New promo: Earn double points on every purchase this week!",
    #     "We have an important update regarding your bank account.",
    #     "Exclusive webinar for premium members happening this weekend.",
    #     "Your electricity bill is now available. Pay before the due date.",
    #     "Don't miss the chance to win a luxury cruise! Register now!",
    #     "This is a friendly reminder for your upcoming doctor’s appointment.",
    #     "Flash sale! All items are 50% off for the next 24 hours.",
    #     "Important notice: Changes to our privacy policy take effect soon.",
    #     "Your account login was attempted from an unknown device. Verify!",
    #     "Hi, hope to see you at the family gathering this Saturday!",
    #     "You've won a free ticket to the concert of the year! Claim now.",
    #     "Congratulations! You’re a lucky winner of a weekend getaway!",
    #     "New course available: Learn Python programming from scratch.",
    #     # "Your password was successfully changed. Contact us if this wasn’t you."
    # ]
    #
    # y = [
    #     1, 0, 0, 0, 0, 0, 1, 0,
    #     1, 0, 0, 0, 1, 0, 0, 1,
    #     0, 0, 0, 0,
    #     0, 0, 1, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 1, 0, 1, 0,
    #     1, 0, 1, 0, 0, 0, 1, 0,
    #     0, 1, 0, 1, 0, 0, 1, 0,
    #     # 1
    # ]  # 1 = spam, 0 = ham
    #
    # return np.array(X), np.array(y)