# Chatbot-printer

Un chatbot de prise en compte automatique de demande d’impression par « ChatBot » et prévoir l’ordonnancement des tâches d’impression.

## Requirement

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install nltk.

```bash
pip3 install nltk numpy torch 
```
## Usage

Il faut entrainer le model avant d'utiliser le ChatBot, pour cela taper:

```bash
python3 train.py
```
Maintenant vous pouvez lancer le ChatBot:

```bash
python3 chatbot.py
```