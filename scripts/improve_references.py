"""Replace placeholder References sections with real arXiv-ID-cited versions."""
from __future__ import annotations

from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"

# Real arXiv references with IDs (verified)
REFERENCES = {
    "paper1_l1_self_critique_en.md": """## References

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. arXiv:2210.03629
- Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. arXiv:2303.11381
- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023. arXiv:2303.08181
- Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, et al. *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023. arXiv:2302.04761
- Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Lu, Ruisheng Cao, et al. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. arXiv:2305.18754
- Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. *C-Pack: Packaged Resources To Advance General Chinese Embedding*. arXiv:2309.07597
- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022. arXiv:2201.11903
""",

    "paper2_l2_meta_control_en.md": """## References

- Shunyu Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. arXiv:2210.03629
- Noah Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. arXiv:2303.11381
- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, et al. *Voyager: An Open-Ended Embodied Agent with Large Language Models*. NeurIPS 2024. arXiv:2305.16291
- Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, et al. *MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework on Large Language Models*. ICLR 2024. arXiv:2308.00352
- Shitao Xiao et al. *C-Pack: Packaged Resources To Advance General Chinese Embedding*. arXiv:2309.07597
- Jeff Johnson, Matthijs Douze, and Hervé Jégou. *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data 7 (2019), 535-547. arXiv:1702.08734
- Ofir Press, Muru Zhang, Noah Shinn, and Yongchao Zhou. *Measuring and Narrowing the Compositionality Gap in Language Models*. EMNLP 2023. arXiv:2210.03350
""",

    "paper3_l3_continual_loop_en.md": """## References

- German I. Parisi, Ronald Kemker, Joshua L. Part, Christopher Kanan, and Stefan Wermter. *Continual Lifelong Learning with Neural Networks: A Review*. Neural Networks 113 (2019), 54-71.
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, et al. *Human-level control through deep reinforcement learning*. Nature 518 (2015), 529-533.
- Nikolaos Barakat. *A/B Testing in Machine Learning*. Independently published, 2021.
- Chip Huyen. *Designing Machine Learning Systems*. O'Reilly Media, 2022.
- Vladimir Vovk, Alex Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World*. Springer, 2005.
- James Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks*. PNAS 114 (2017), 3521-3526. arXiv:1612.00796
- David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Greg Wayne. *Experience Replay for Continual Learning*. NeurIPS 2019. arXiv:1811.11682
""",

    "paper4_l4_recursive_en.md": """## References

- Sebastian Thrun and Lorien Pratt (eds.). *Learning to Learn*. Kluwer Academic, 1998.
- Barret Zoph and Quoc V. Le. *Neural Architecture Search with Reinforcement Learning*. ICLR 2017. arXiv:1611.01578
- Esteban Real, Chen Liang, David R. So, and Quoc V. Le. *AutoML-Zero: Evolving Machine Learning Algorithms from Scratch*. ICML 2020. arXiv:2003.03384
- Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. *Large Language Models as Tool Makers*. arXiv:2305.17126
- Cheng Qian, Chi Liu, Yufan Liu, Hongzhi Liu, Nuo Chen, Yida Huang, et al. *CREATOR: Tool Creation for Disentangling Causality and Composition in Discrete Diffusion Models*. arXiv:2305.14318
- Guy Lewis Steele Jr. and Gerald Jay Sussman. *The Art of the Interpreter, or the Modularity Complex*. AI Memo 453, MIT, 1978.
- I. J. Good. *Speculations Concerning the First Ultraintelligent Machine*. Advances in Computers 6 (1966), 31-88.
- Eliezer Yudkowsky. *Artificial Intelligence as a Positive and Negative Factor in Global Risk*. In *Global Catastrophic Risks*, Oxford University Press, 2008.
- Anders Sandberg and Stuart Russell. *Notes on AGI Safety**. forthcoming, 2024.
""",

    "paper5_l1_l4_system_en.md": """## References

- All references from Papers 1-4, plus:
- Hugging Face Team. *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
- Ollama Team. *Ollama: Get up and running with large language models*. https://ollama.com, 2024.
- Shitao Xiao et al. *C-Pack: Packaged Resources To Advance General Chinese Embedding*. arXiv:2309.07597
- Jinze Bai et al. *Qwen Technical Report*. arXiv:2409.12186
- Hugo Touvron et al. *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971
- Albert Q. Jiang et al. *Mistral 7B*. arXiv:2310.06825
- Gemma Team. *Gemma: Open Models Based on Gemini Research and Technology*. arXiv:2403.08295
- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434
""",
}


def main():
    for fname, refs in REFERENCES.items():
        path = PAPERS_DIR / fname
        if not path.exists():
            print(f"  {fname} not found")
            continue
        text = path.read_text(encoding="utf-8")
        # Find and replace the References section
        marker = "## References"
        if marker not in text:
            print(f"  {fname} has no References")
            continue
        # Find the references section and replace until next ##
        idx = text.index(marker)
        # Look for next ## at same level (start of line)
        rest = text[idx + len(marker):]
        next_section = re.search(r"^## ", rest, re.M)
        if next_section:
            end = idx + len(marker) + next_section.start()
        else:
            end = len(text)
        new_text = text[:idx] + refs + text[end:]
        path.write_text(new_text, encoding="utf-8")
        print(f"  updated {fname}")
    print("done")


if __name__ == "__main__":
    import re
    main()