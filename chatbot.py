from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import unicodedata
import json
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords')

# Dados atualizados com novas recomendações de armações
data = pd.DataFrame({
    'id': [1,2,3,4,5,6,7,8,9,10,11,12],
    'title': [
        "Lentes de Contato Gelatinosas",
        "Lentes Oftálmicas Normais",
        "Entrega Expressa",
        "Entrega Padrão",
        "Armação Acetato Clássica",
        "Armação Metalizada Moderna",
        "Formas de Pagamento - Cartão e Boleto",
        "Formas de Pagamento - Pix e Transferência",
        "Exame de Vista Básico - Clínica VisionPlus",
        "Exame Completo - Clínica Saúde Visual",
        "Armação Clássica Acetato Preto",
        "Armação Metalizada Prata Moderna"
    ],
    'description': [
        "Lentes de contato gelatinosas descartáveis para uso diário.",
        "Lentes para óculos normais com alta precisão óptica.",
        "Entrega rápida em até 24 horas para sua comodidade.",
        "Entrega padrão com prazo de 3 a 5 dias úteis.",
        "Armação de acetato em design clássico e elegante.",
        "Armação metálica com acabamento moderno e resistente.",
        "Aceitamos cartão de crédito, boleto bancário para sua facilidade.",
        "Aceitamos Pix e transferências bancárias com confirmação imediata.",
        "Exame básico de visão realizado na clínica VisionPlus, licenciada e certificada.",
        "Exame completo com equipamentos modernos na clínica Saúde Visual, centros licenciados.",
        "Armação de acetato preto com design clássico e confortável.",
        "Armação metálica prata com visual moderno e resistente."
    ],
    'category': [
        "lentes", "lentes",
        "entrega", "entrega",
        "armação", "armação",
        "pagamento", "pagamento",
        "exame", "exame",
        "armação", "armação"
    ]
})

# FAQ atualizadas, incluído tópico específico recomendação de armações
faq_perguntas_variadas = [
    # Armação
    ("armação ideal", "Para escolher a armação ideal, leve em conta seu formato de rosto, estilo pessoal e conforto. Armações redondas combinam com rostos quadrados, enquanto armações retangulares ficam bem em rostos arredondados. Cores e materiais influenciam no estilo. Posso ajudar a encontrar a melhor opção para você."),
    ("qual armacao escolher", "Para escolher a armação ideal, leve em conta seu formato de rosto, estilo pessoal e conforto. Armações redondas combinam com rostos quadrados, enquanto armações retangulares ficam bem em rostos arredondados. Cores e materiais influenciam no estilo. Posso ajudar a encontrar a melhor opção para você."),
    ("como escolher armacao", "Para escolher a armação ideal, leve em conta seu formato de rosto, estilo pessoal e conforto. Armações redondas combinam com rostos quadrados, enquanto armações retangulares ficam bem em rostos arredondados. Cores e materiais influenciam no estilo. Posso ajudar a encontrar a melhor opção para você."),
    ("informações sobre armação", "Oferecemos armações de acetato clássico e metalizadas modernas, para diferentes estilos e conforto. Posso ajudar a encontrar a armação ideal para você."),
    ("quero saber sobre armação", "Oferecemos armações de acetato clássico e metalizadas modernas, para diferentes estilos e conforto. Posso ajudar a encontrar a armação ideal para você."),
    ("quais armações você recomenda", "Recomendo armações de acetato clássico para conforto e estilo atemporal, e armações metalizadas para um visual moderno e sofisticado."),
    ("melhores armações", "As melhores armações dependem do seu estilo e formato do rosto. Posso te ajudar a encontrar a opção ideal."),
    ("armações recomendadas", "Temos armações para diversos gostos e necessidades, incluindo acetato e metal. Gostaria de ver as opções?"),
    # Lentes
    ("informações sobre lentes", "Temos lentes de contato gelatinosas descartáveis para uso diário, lentes oftálmicas normais com alta precisão óptica. Posso ajudar com detalhes dos tipos, cuidados e recomendações."),
    ("quero saber sobre lentes", "Temos lentes de contato gelatinosas descartáveis para uso diário, lentes oftálmicas normais com alta precisão óptica. Posso ajudar com detalhes dos tipos, cuidados e recomendações."),
    ("informacao sobre lentes", "Temos lentes de contato gelatinosas descartáveis para uso diário, lentes oftálmicas normais com alta precisão óptica. Posso ajudar com detalhes dos tipos, cuidados e recomendações."),
    ("como colocar lente", "Para colocar a lente de contato, lave bem as mãos, tire a lente do estojo, posicione no dedo indicador e cuidadosamente coloque sobre o olho."),
    ("cuidados lente", "Lave sempre as mãos antes de manusear lentes, use soluções recomendadas para limpeza e guarde as lentes em estojo limpo e seco."),
    ("posso dormir com lente", "Dormir com lentes de contato não é recomendado, pois pode causar irritação e infecções oculares."),
    # Entrega
    ("informações sobre entrega", "Oferecemos entrega expressa em até 24 horas e entrega padrão com prazo de 3 a 5 dias úteis. Posso ajudar com detalhes sobre prazos e modalidades."),
    ("prazo entrega", "Oferecemos entrega expressa em até 24 horas e entrega padrão com prazo de 3 a 5 dias úteis. Posso ajudar com detalhes sobre prazos e modalidades."),
    ("quero saber sobre entrega", "Oferecemos entrega expressa em até 24 horas e entrega padrão com prazo de 3 a 5 dias úteis. Posso ajudar com detalhes sobre prazos e modalidades."),
    # Pagamento
    ("formas de pagamento", "Aceitamos cartão de crédito, boleto bancário, Pix e transferências bancárias para sua comodidade."),
    ("quero saber formas de pagamento", "Aceitamos cartão de crédito, boleto bancário, Pix e transferências bancárias para sua comodidade."),
    ("informações sobre pagamento", "Aceitamos cartão de crédito, boleto bancário, Pix e transferências bancárias para sua comodidade."),
    # Exame
    ("informações sobre exame", "Realizamos exames básicos e completos em nossas clínicas licenciadas VisionPlus e Saúde Visual. É necessário agendar previamente."),
    ("exame vista", "Realizamos exames básicos e completos em nossas clínicas licenciadas VisionPlus e Saúde Visual. É necessário agendar previamente."),
    ("quero saber sobre exame", "Realizamos exames básicos e completos em nossas clínicas licenciadas VisionPlus e Saúde Visual. É necessário agendar previamente."),
    # Outros
    ("lente descartavel", "As lentes descartáveis devem ser usadas pelo tempo indicado e descartadas após o uso para garantir segurança.")
]

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_perguntas = [item[0] for item in faq_perguntas_variadas]
faq_respostas = [item[1] for item in faq_perguntas_variadas]

faq_embeddings = model.encode(faq_perguntas, convert_to_tensor=True)

stopwords_pt = stopwords.words('portuguese')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
product_vectors = vectorizer.fit_transform(data['description'])

estado_usuario = {
    "fase": "inicial",
    "preferencias": "",
    "feedback": [],
    "topico_atual": None,
    "ultima_resposta": None
}

def remove_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def responder_faq_semantico(mensagem, threshold=0.4):
    msg_norm = remove_acentos(mensagem.lower())
    mensagem_emb = model.encode([msg_norm], convert_to_tensor=True)
    similaridades = util.pytorch_cos_sim(mensagem_emb, faq_embeddings)[0]
    melhor_indice = int(np.argmax(similaridades.cpu()))
    melhor_score = float(similaridades[melhor_indice].cpu())
    print(f"DEBUG: Similaridade encontrada = {melhor_score}")
    if melhor_score >= threshold:
        return faq_respostas[melhor_indice]
    return None

def recomendar(preferencias_usuario, top_n=3):
    preferencias_vetorizadas = vectorizer.transform([preferencias_usuario])
    similaridade = cosine_similarity(preferencias_vetorizadas, product_vectors)
    indices = np.argsort(similaridade[0])[::-1][:top_n]
    recomendacoes = data.iloc[indices]
    return recomendacoes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        raw_data = request.get_data()
        decoded_data = raw_data.decode('utf-8')
        data_json = json.loads(decoded_data)
        entrada = data_json.get('mensagem', '').strip()
    except Exception as e:
        return jsonify({"resposta": f"Erro ao processar requisição JSON: {str(e)}"}), 400

    global estado_usuario

    entrada_lower = entrada.lower()
    entrada_norm = remove_acentos(entrada_lower)

    comandos_encerrar = ["tchau", "finalizar", "encerrar", "sair", "até logo"]
    if any(x in entrada_lower for x in comandos_encerrar):
        resposta = "Obrigado pelo contato! Caso precise, estarei aqui para ajudar. Até breve!"
        estado_usuario["fase"] = "inicial"
        estado_usuario["preferencias"] = ""
        estado_usuario["feedback"] = []
        estado_usuario["topico_atual"] = None
        estado_usuario["ultima_resposta"] = None
        return jsonify({"resposta": resposta})

    saudacoes = ["olá", "ola", "oi", "bom dia", "boa tarde", "boa noite", "saudações"]
    if any(remove_acentos(s) in entrada_norm for s in saudacoes):
        if estado_usuario.get("topico_atual") is not None:
            resposta = "Por favor, poderia reformular ou redigir melhor sua dúvida para que eu possa ajudar?"
            estado_usuario['ultima_resposta'] = resposta
            return jsonify({"resposta": resposta})
        else:
            resposta = "Olá! Como posso ajudar você hoje? Pode perguntar sobre lentes, entrega, armação, pagamento ou exames."
            estado_usuario['ultima_resposta'] = resposta
            return jsonify({"resposta": resposta})

    recomendacoes_palavras = ["recomenda", "qual você recomenda", "qual vc recomenda", "qual seu conselho", "qual me recomenda"]
    if any(palavra in entrada_lower for palavra in recomendacoes_palavras):
        topico = estado_usuario.get("topico_atual")
        if topico:
            recomendados = recomendar(topico)
            items = ", ".join(recomendados['title'].tolist())
            resposta = f"Baseado no tópico {topico}, aqui estão as recomendações: {items}. Deseja ver mais opções? (sim/não)"
        else:
            resposta = "Por favor, informe qual tópico deseja recomendações, como armação, lentes, entrega, pagamento ou exame."
        estado_usuario['ultima_resposta'] = resposta
        return jsonify({"resposta": resposta})

    if estado_usuario["fase"] == "inicial":
        if len(entrada_lower.split()) < 2:
            resposta = "Por favor, me diga como posso ajudar com mais detalhes, pode ser sobre lentes, armação, entrega, pagamento ou exames."
            estado_usuario['ultima_resposta'] = resposta
            return jsonify({"resposta": resposta})
        else:
            resposta = ("Olá! Posso ajudar com lentes (contato ou normais), entrega, armações, "
                       "formas de pagamento, exames e clínicas. Sobre o que gostaria de saber?")
            estado_usuario["fase"] = "coletando_preferencias"
            estado_usuario['ultima_resposta'] = resposta
            return jsonify({"resposta": resposta})

    resposta_faq = responder_faq_semantico(entrada)
    if resposta_faq:
        if estado_usuario.get('ultima_resposta') == resposta_faq:
            resposta = "Desculpe, não entendi muito bem. Poderia reformular ou repetir, por favor?"
        else:
            resposta = resposta_faq
        estado_usuario['ultima_resposta'] = resposta
        return jsonify({"resposta": resposta})

    resposta = ""

    if estado_usuario["fase"] == "coletando_preferencias":
        estado_usuario["preferencias"] = entrada_lower

        if any(x in entrada_lower for x in ["armação", "óculos", "oculos"]):
            resposta = ("Você está interessado em armações? Temos armações de acetato clássico e metalizadas modernas. "
                       "Quer que eu te mostre algumas opções? (responda sim ou não)")
            estado_usuario["fase"] = "confirmando"
            estado_usuario["topico_atual"] = "armação"

        elif any(x in entrada_lower for x in ["lente", "contato", "lentes"]):
            resposta = ("Você quer informações sobre lentes? Temos lentes de contato descartáveis e lentes oftálmicas normais. "
                       "Quer ver as opções? (responda sim ou não)")
            estado_usuario["fase"] = "confirmando"
            estado_usuario["topico_atual"] = "lentes"

        elif any(x in entrada_lower for x in ["entrega", "prazo", "frete"]):
            resposta = ("Sobre entregas, oferecemos entrega expressa em 24h e entrega padrão em 3 a 5 dias úteis. "
                       "Quer ver as opções? (responda sim ou não)")
            estado_usuario["fase"] = "confirmando"
            estado_usuario["topico_atual"] = "entrega"

        elif any(x in entrada_lower for x in ["pagamento", "forma", "cartão", "boleto", "pix"]):
            resposta = ("Temos várias formas de pagamento, como cartão de crédito, boleto, Pix e transferências bancárias. "
                       "Quer ver as opções? (responda sim ou não)")
            estado_usuario["fase"] = "confirmando"
            estado_usuario["topico_atual"] = "pagamento"

        elif any(x in entrada_lower for x in ["exame", "vista", "clinica", "clínica"]):
            resposta = ("Oferecemos exames básicos e completos em clínicas licenciadas como VisionPlus e Saúde Visual. "
                       "Quer ver as opções? (responda sim ou não)")
            estado_usuario["fase"] = "confirmando"
            estado_usuario["topico_atual"] = "exame"

        else:
            recomendados = recomendar(entrada_lower)
            items = ", ".join(recomendados['title'].tolist())
            if estado_usuario.get('ultima_resposta') == f"Aqui estão algumas sugestões para você: {items}. Deseja mais opções ou quer refinar a busca?":
                resposta = "Desculpe, não consegui entender bem. Poderia reformular sua pergunta, por favor?"
            else:
                resposta = f"Aqui estão algumas sugestões para você: {items}. Deseja mais opções ou quer refinar a busca?"
            estado_usuario["fase"] = "esperando_feedback"

    elif estado_usuario["fase"] == "confirmando":
        topico = estado_usuario.get("topico_atual", None)

        if any(x in entrada_lower for x in ["sim", "s", "quero", "gostaria", "mais"]):
            if topico:
                recomendados = recomendar(topico)
                items = ", ".join(recomendados['title'].tolist())
                resposta = f"Aqui estão as opções para {topico}: {items}. Deseja ver mais opções desse tópico? (sim/não)"
            else:
                resposta = "Sobre qual assunto você gostaria de receber recomendações? Por exemplo, armação, lente, entrega, pagamento ou exame."
            estado_usuario["fase"] = "confirmando"
        else:
            resposta = "Tudo bem! Se quiser saber sobre outro assunto, é só falar."
            estado_usuario = {"fase": "inicial", "preferencias": "", "feedback": [], "topico_atual": None}

    elif estado_usuario["fase"] == "esperando_feedback":
        if "mais" in entrada_lower:
            recomendados = recomendar(estado_usuario["preferencias"], top_n=5)
            items = ", ".join(recomendados['title'].tolist())
            resposta = f"Aqui estão mais opções: {items}. Deseja continuar refinando?"
        elif "refinar" in entrada_lower:
            resposta = "Claro! O que você gostaria de mudar? Pode citar lentes, entrega, armação, pagamento ou exames."
            estado_usuario["fase"] = "refinando"
        else:
            resposta = "Obrigado por usar nosso serviço! Se precisar, estarei aqui para ajudar."
            estado_usuario = {"fase": "inicial", "preferencias": "", "feedback": [], "topico_atual": None}

    elif estado_usuario["fase"] == "refinando":
        estado_usuario["feedback"].append(entrada_lower)
        texto_ajustado = estado_usuario["preferencias"] + " " + " ".join(estado_usuario["feedback"])
        recomendados = recomendar(texto_ajustado)
        items = ", ".join(recomendados['title'].tolist())
        resposta = f"Aqui estão suas recomendações refinadas: {items}. Precisa de mais alguma coisa?"
        estado_usuario["fase"] = "esperando_feedback"

    else:
        resposta = "Desculpe, não entendi. Por favor, tente novamente."

    estado_usuario['ultima_resposta'] = resposta
    return jsonify({"resposta": resposta})

if __name__ == '__main__':
    app.run(debug=True)
