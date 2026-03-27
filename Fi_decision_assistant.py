#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Decision Assistant
Based on Qwen3-0.6B fine-tuned with LoRA on financial data,
providing market interpretation, risk analysis, strategy reasoning, and scenario analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os

# Financial scenario system prompts
FINANCIAL_PROMPTS = {
    "market_interpretation": "You are a financial market analyst. Interpret the given market data, price action, sentiment indicators, or sector rotation patterns and explain what they signal.",
    "risk_analysis": "You are a financial risk analyst. Evaluate the risk profile, downside exposure, or credit risk based on the given portfolio or position data.",
    "strategy_reasoning": "You are a trading strategist. Based on the given market conditions and position data, suggest appropriate trading strategies, position management, or hedging approaches.",
    "scenario_analysis": "You are a financial scenario analyst. Given the current market conditions, project what could happen next under various assumptions and explain the likely consequences.",
    "general": "You are a financial expert. Answer the user's financial question with a clear, well-reasoned response covering relevant concepts, practical advice, and important considerations.",
}

# Scenario menu
FINANCIAL_SCENARIOS = {
    "1": "Market Interpretation",
    "2": "Risk Analysis",
    "3": "Strategy Reasoning",
    "4": "Scenario Analysis",
    "5": "General Financial Q&A",
}

# Sample questions for each scenario
SAMPLE_QUESTIONS = {
    "market_interpretation": [
        "What does this market behavior indicate?\nStock: TSLA; 5-day return: -7.2%; rolling volatility increased 18%; trend: downward.",
        "Interpret the recent price action.\nStock: NVDA; 5-day return: +12.3%; rolling volatility increased 25%; trend: upward.",
        "What does the current market data suggest about investor sentiment?\nS&P 500 down 2.1% this week; VIX jumped from 15 to 24; bond yields fell 12 bps.",
    ],
    "risk_analysis": [
        "Assess the risk profile.\nStock: TSLA; 5-day return: -7.2%; rolling volatility increased 18%; trend: downward.",
        "Evaluate the portfolio risk.\nPortfolio: 60% equities, 40% bonds; equity portion down 5.3% this month; portfolio beta: 1.15.",
        "What are the key risks facing this position?\nStock: AMZN; earnings report in 3 days; implied volatility at 45% vs 30% historical.",
    ],
    "strategy_reasoning": [
        "How might a short-term trader respond?\nStock: AAPL; momentum weakened after a strong rally and volatility rose moderately.",
        "What trading strategy would suit this environment?\nMarket: sideways range for 3 weeks; VIX at 13; S&P 500 between 4,450 and 4,520.",
        "What hedging strategy makes sense here?\nPortfolio heavily concentrated in tech stocks; correlation among holdings is 0.85.",
    ],
    "scenario_analysis": [
        "If volatility rises further and returns remain negative, what could happen next?\nStock: META; negative returns persisted for 4 days.",
        "What could happen if interest rates rise while corporate earnings weaken?\n10-year yield up 50 bps; earnings estimates revised down 3%.",
        "What happens to growth stocks if the Fed signals more rate hikes?\nGrowth ETF (VUG) already down 6% from peak; Nasdaq P/E at 28x.",
    ],
    "general": [
        "Where should I be investing my money?",
        "What is the difference between a Roth IRA and a traditional IRA?",
        "How does compound interest work and why does it matter?",
    ],
}


class FinancialAssistant:
    def __init__(self, checkpoint_path="./output/Qwen3-0.6B-financial/checkpoint-2258"):
        self.checkpoint_path = checkpoint_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []

    def _select_device_and_dtype(self):
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32

    def load_model(self):
        print("Loading financial assistant model...")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model path not found: {self.checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=self.dtype,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded! Device: {self.device}")

    def predict(self, messages, max_new_tokens=512):
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response

    def ask_question(self, question, scenario_type="market_interpretation", max_tokens=512):
        if scenario_type not in FINANCIAL_PROMPTS:
            scenario_type = "market_interpretation"

        messages = [
            {"role": "system", "content": FINANCIAL_PROMPTS[scenario_type]},
            {"role": "user", "content": question}
        ]

        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_type,
            "question": question,
            "response": None
        })

        response = self.predict(messages, max_new_tokens=max_tokens)
        self.conversation_history[-1]["response"] = response

        return response

    def show_scenarios(self):
        print("\nFinancial Decision Assistant - Scenarios:")
        print("=" * 50)
        for key, value in FINANCIAL_SCENARIOS.items():
            print(f"  {key}. {value}")
        print("=" * 50)

    def show_sample_questions(self, scenario_type):
        if scenario_type in SAMPLE_QUESTIONS:
            scenario_label = FINANCIAL_SCENARIOS.get(
                str(list(FINANCIAL_PROMPTS.keys()).index(scenario_type) + 1), scenario_type
            )
            print(f"\nSample questions - {scenario_label}:")
            print("-" * 40)
            for i, question in enumerate(SAMPLE_QUESTIONS[scenario_type], 1):
                print(f"  {i}. {question}")
            print("-" * 40)

    def interactive_mode(self):
        print("\nFinancial Decision Assistant started!")
        print("Type 'help' for help, 'quit' to exit.")

        while True:
            try:
                self.show_scenarios()

                scenario_choice = input("\nSelect a scenario (1-5): ").strip()
                if scenario_choice == 'quit':
                    break
                elif scenario_choice == 'help':
                    self.show_help()
                    continue
                elif scenario_choice not in FINANCIAL_SCENARIOS:
                    print("Invalid choice, please try again.")
                    continue

                scenario_type = list(FINANCIAL_PROMPTS.keys())[int(scenario_choice) - 1]

                self.show_sample_questions(scenario_type)

                question = input(f"\nEnter your {FINANCIAL_SCENARIOS[scenario_choice]} question: ").strip()
                if not question:
                    print("Question cannot be empty.")
                    continue

                print("\nAnalyzing...")
                start_time = time.time()

                response = self.ask_question(question, scenario_type)

                elapsed_time = time.time() - start_time

                print(f"\nResponse ({elapsed_time:.2f}s):")
                print("=" * 60)
                print(response)
                print("=" * 60)

                continue_choice = input("\nContinue? (y/n): ").strip().lower()
                if continue_choice in ['n', 'no']:
                    break

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

    def show_help(self):
        print("\nFinancial Decision Assistant - Help:")
        print("=" * 50)
        print("1. Select a financial scenario (1-4)")
        print("2. Enter your question with relevant market data")
        print("3. Get an AI-powered financial analysis")
        print("\nNote:")
        print("- This assistant provides analysis for reference only")
        print("- Not financial advice — always do your own research")
        print("- Type 'quit' to exit")
        print("=" * 50)

    def save_conversation(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_conversation_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

        print(f"Conversation saved to: {filename}")

    def batch_questions(self, questions_file):
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)

            print(f"Processing {len(questions)} questions...")

            results = []
            for i, q in enumerate(questions, 1):
                print(f"\nProcessing {i}/{len(questions)}...")
                response = self.ask_question(
                    q.get('question', ''),
                    q.get('scenario', 'market_interpretation'),
                    q.get('max_tokens', 512)
                )

                results.append({
                    "question": q.get('question', ''),
                    "scenario": q.get('scenario', 'market_interpretation'),
                    "response": response
                })

            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Batch processing complete! Results saved to: {output_file}")

        except Exception as e:
            print(f"Batch processing failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Financial Decision Assistant - Qwen3-0.6B LoRA fine-tuned")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="./output/Qwen3-0.6B-financial/checkpoint-2258",
                        help="Model checkpoint path")
    parser.add_argument("--question", "-q", type=str,
                        help="Ask a single question (use with --scenario)")
    parser.add_argument("--scenario", "-s", type=str,
                        default="market_interpretation",
                        choices=list(FINANCIAL_PROMPTS.keys()),
                        help="Financial scenario type")
    parser.add_argument("--max-tokens", "-m", type=int,
                        default=512,
                        help="Maximum generation tokens")
    parser.add_argument("--batch", "-b", type=str,
                        help="Batch process questions file (JSON)")
    parser.add_argument("--save-history", action="store_true",
                        help="Save conversation history")

    args = parser.parse_args()

    assistant = FinancialAssistant(args.checkpoint)
    assistant.load_model()

    if args.batch:
        assistant.batch_questions(args.batch)
    elif args.question:
        print("Financial Assistant Response:")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        assistant.interactive_mode()

    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()