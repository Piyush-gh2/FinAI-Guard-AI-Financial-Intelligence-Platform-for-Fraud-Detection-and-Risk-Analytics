from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_report(transaction):

    prompt = f"""
    Financial Investigation Report

    Transaction Amount: {transaction['Amount']}
    Location: {transaction['Location']}
    Device: {transaction['Device']}

    Analyze if this transaction looks suspicious.
    """

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        inputs,
        max_length=120,
        num_return_sequences=1
    )

    report = tokenizer.decode(output[0], skip_special_tokens=True)

    return report