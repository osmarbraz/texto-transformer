# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.util.utiltexto import getIndexTokenTexto, getTextoLista, limpezaTexto, removeTags, tamanhoTexto, contaItensLista, truncaJanela, getJanelaLista

# Objeto de logger
logger = logging.getLogger(__name__)

class TestUtilTexto(unittest.TestCase):
        
    # Testes limpeza
    def test_limpeza(self):
        logger.info("Testando o limpeza")
        
        # Valores de entrada
        texto = "   Qual o \nsabor   do  \n  sorvete   ??????????   "
        
        # Valores de saída
        saida = limpezaTexto(texto)
        
        # Valores esperados
        saidaEsperada = "Qual o sabor do sorvete ?"
                
        self.assertEqual(saida, saidaEsperada)
        
    # Testes getTextoLista
    def test_getTextoLista(self):
        logger.info("Testando o getTextoLista")
        
        # Valores de entrada
        texto = ['um','dois']
        
        # Valores de saída
        saida = getTextoLista(texto)
        
        # Valores esperados
        saidaEsperada = "umdois"
                
        self.assertEqual(saida, saidaEsperada)        
        
    # Testes removeTags
    def test_removeTags(self):
        logger.info("Testando o removeTags")
        
        # Valores de entrada
        texto = '<html><body>texto</body></html>'
        
        # Valores de saída
        saida = removeTags(texto)
        
        # Valores esperados
        saidaEsperada = "texto"
                
        self.assertEqual(saida, saidaEsperada)
        
    # Testes tamanhoTexto
    def test_tamanhoTexto(self):
        logger.info("Testando o tamanhoTexto")
        
        # Valores de entrada
        texto1 = ""
        texto2 = "manga"
        texto3 = []
        texto4 = [["manga","banana"]]
        texto5 = [["manga","banana"],["uva","laranja"]]
        texto6 = {'lista1' :[["manga","banana"]]}        
        texto7 = [{'lista1' :[["manga","banana"]], 
                   'lista2' : [["uva","laranja"]]}]

        # Avalia a saida do método
        self.assertEqual(tamanhoTexto(texto1), 0)
        self.assertEqual(tamanhoTexto(texto2), 5)
        self.assertEqual(tamanhoTexto(texto3), 0)
        self.assertEqual(tamanhoTexto(texto4), 2)
        self.assertEqual(tamanhoTexto(texto5), 4)
        self.assertEqual(tamanhoTexto(texto6), 1)
        self.assertEqual(tamanhoTexto(texto7), 2)

    # Testes getIndexTokenTexto
    def test_tamanhoTexto(self):
        logger.info("Testando o getIndexTokenTexto")
        
        # Valores de entrada
        lista_tokens = ['Depois', 'de', 'roubar', 'o', 'co', '##fre', 'do', 'banco', ',', 'o', 'lad', '##rão', 'de', 'banco', 'foi', 'visto', 'sentado', 'no', 'banco', 'da', 'praça', 'central', '.']
        token = "banco"
        # O token "banco" se encontra nas posições  7, 13 e 18
        
        # Valores de saída
        idx_tokens = getIndexTokenTexto(lista_tokens, token)
                       
        # Avalia a saida do método       
        self.assertEqual(len (idx_tokens), 3)
        self.assertEqual(idx_tokens[0], 7)
        self.assertEqual(idx_tokens[1], 13)
        self.assertEqual(idx_tokens[2], 18)
        
    # Testes contaItensLista
    def test_contaItensLista(self):
        logger.info("Testando o contaItensLista")
        
        # Valores de entrada
        lista = [['P101', 'p102', 'p103', 'p104'], ['p105', 'p106', 'p107', 'p108']]
                              
        # Avalia a saida do método       
        self.assertEqual(contaItensLista(lista), 8)

    # Testes truncaJanelaCentro
    def test_truncaJanelaCentro(self):
        logger.info("Testando o truncaJanelaCentro")
        
        # Valores de entrada
        lista_janela = [['P101', 'p102', 'p103', 'p104', '.'], ['p201', 'p202', 'p203', 'p204', '.'], ['p301', 'p302', 'p303', '3204',  '.']]
        maximo_itens = 10
        lista_centro_janela = [1]

        lista_saida_janela = truncaJanela(lista_janela, maximo_itens, lista_centro_janela)
        
        lista_esperada_janela = [['p104', '.'], ['p201', 'p202', 'p203', 'p204', '.'], ['p301', 'p302', 'p303']]
                              
        # Avalia a saida do método       
        self.assertListEqual(lista_saida_janela, lista_esperada_janela)

    # Testes truncaJanelaEsquerda
    def test_truncaJanelaEsquerda(self):
        logger.info("Testando o truncaJanelaEsquerda")
        
        # Valores de entrada
        lista_janela = [['P101', 'p102', 'p103', 'p104', '.'], ['p201', 'p202', 'p203', 'p204', '.'], ['p301', 'p302', 'p303', '3204',  '.']]
        maximo_itens = 9
        lista_centro_janela = [0]

        lista_saida_janela = truncaJanela(lista_janela, maximo_itens, lista_centro_janela)
        
        lista_esperada_janela = [['P101', 'p102', 'p103', 'p104', '.'], ['p201', 'p202', 'p203', 'p204'], []]
                              
        # Avalia a saida do método       
        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
    
    # Testes truncaJanelaDireita
    def test_truncaJanelaDireita(self):
        logger.info("Testando o truncaJanelaDireita")
        
        # Valores de entrada
        lista_janela = [['P101', 'p102', 'p103', 'p104', '.'], ['p201', 'p202', 'p203', 'p204', '.'], ['p301', 'p302', 'p303', '3204',  '.']]
        maximo_itens = 9
        lista_centro_janela = [2]

        lista_saida_janela = truncaJanela(lista_janela, maximo_itens, lista_centro_janela)
        
        lista_esperada_janela = [[], ['p202', 'p203', 'p204', '.'], ['p301', 'p302', 'p303', '3204', '.']]
                              
        # Avalia a saida do método       
        self.assertListEqual(lista_saida_janela, lista_esperada_janela)    
                
    # Testes getJanelaSentenca3DocumentoMaior
    def test_getJanelaSentenca3DocumentoMaior(self):
        logger.info("Testando o getJanelaSentenca3 com lista de documento maior que a janela ")
        
        # Valores de entrada
        tamanho_janela = 3
        trunca_janela = 30

        # Lista de sentenças
        lista_sentenca = ['P101 p102 p103 p104 p105 p106 p107 p108 p109 p110 p111 p112 p113 p114 p115 p116 p117 p118 p119 p120 .',
                        'p201 p202 p203 p204 p205 p206 p207 p208 p209 p210 p211 p212 p213 p214 p215 p216 p217 p218 p219 p220 p221 p222 p223 p224 .',
                        'p301 p302 p303 p304 p305 p306 p307 p308 p309 p310 p311 p312 p313 p314 p315 p316 p317 p318 p319 p320 p321 p322 p323 p324 p325 .',
                        'p401 p402 p403 p404 p405 p406 p407 p408 p409 p410 p411 p412 p413 p414 p415 p416 p417 p418 p419 p420 p421 p422 p423 p424 p425 p426 .',
                        'p501 p502 p503 p504 p505 p506 p507 p508 p509 p510 p511 p512 p513 p514 p515 p516 p517 p518 p519 p520 p521 p522 p523 p524 p525 .',
                        'p601 p602 p603 p604 p605 p606 p607 p608 p609 p610 p611 p612 p613 p614 p615 p616 p617 p618 p619 p620 p621 p622 p623 p624 .',
                        'p701 p702 p703 p704 p705 p706 p707 p708 p709 p710 p711 p712 p713 p714 p715 p716 p717 p718 p719 p720 p721 p722 p723 .']

        # Divide a sentença em palavras pelos espaços.
        lista = []
        for i, sentenca in enumerate(lista_sentenca):
            # Divide a sentença pelos espaços em branco.
            lista.append(sentenca.split(' '))
        
        # Lista das janelas para comparar os elementos.
        lista_saida_janela = []
        lista_saida_conta_itens = []
        lista_saida_centro = []
        lista_saida_indice = []
        for i, x in enumerate(lista):
            lista_janela, string_janela, lista_indice_janela, centro_janela = getJanelaLista(lista, tamanho_janela, i, trunca_janela)
            lista_saida_janela.append(lista_janela)
            lista_saida_conta_itens.append(contaItensLista(lista_janela))
            lista_saida_centro.append(centro_janela)
            lista_saida_indice.append(lista_indice_janela)
        
        lista_esperada_janela = [[['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', 'p111', 'p112', 'p113', 'p114', 'p115', 'p116', 'p117', 'p118', 'p119', 'p120', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209']], [['p120', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', 'p211', 'p212', 'p213', 'p214', 'p215', 'p216', 'p217', 'p218', 'p219', 'p220', 'p221', 'p222', 'p223', 'p224', '.'], ['p301', 'p302', 'p303']], [['p224', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', 'p311', 'p312', 'p313', 'p314', 'p315', 'p316', 'p317', 'p318', 'p319', 'p320', 'p321', 'p322', 'p323', 'p324', 'p325', '.'], ['p401', 'p402']], [['.'], ['p401', 'p402', 'p403', 'p404', 'p405', 'p406', 'p407', 'p408', 'p409', 'p410', 'p411', 'p412', 'p413', 'p414', 'p415', 'p416', 'p417', 'p418', 'p419', 'p420', 'p421', 'p422', 'p423', 'p424', 'p425', 'p426', '.'], ['p501', 'p502']], [['p426', '.'], ['p501', 'p502', 'p503', 'p504', 'p505', 'p506', 'p507', 'p508', 'p509', 'p510', 'p511', 'p512', 'p513', 'p514', 'p515', 'p516', 'p517', 'p518', 'p519', 'p520', 'p521', 'p522', 'p523', 'p524', 'p525', '.'], ['p601', 'p602']], [['p525', '.'], ['p601', 'p602', 'p603', 'p604', 'p605', 'p606', 'p607', 'p608', 'p609', 'p610', 'p611', 'p612', 'p613', 'p614', 'p615', 'p616', 'p617', 'p618', 'p619', 'p620', 'p621', 'p622', 'p623', 'p624', '.'], ['p701', 'p702', 'p703']], [['p620', 'p621', 'p622', 'p623', 'p624', '.'], ['p701', 'p702', 'p703', 'p704', 'p705', 'p706', 'p707', 'p708', 'p709', 'p710', 'p711', 'p712', 'p713', 'p714', 'p715', 'p716', 'p717', 'p718', 'p719', 'p720', 'p721', 'p722', 'p723', '.']]]
        lista_esperada_conta_itens = [30, 30, 30, 30, 30, 30, 30]
        lista_esperada_centro = [[0], [1], [1], [1], [1], [1], [1]]
        lista_esperada_indice = [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6]]

        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
        self.assertListEqual(lista_saida_conta_itens, lista_esperada_conta_itens)        
        self.assertListEqual(lista_saida_centro, lista_esperada_centro)
        self.assertListEqual(lista_saida_indice, lista_esperada_indice)
        
    # Testes getJanelaSentenca3DocumentoIgualJanela
    def test_getJanelaSentenca3DocumentoIgualJanela(self):
        logger.info("Testando o getJanelaSentenca3 com lista de documento igual a janela ")
        
        # Valores de entrada
        tamanho_janela = 3
        trunca_janela = 30

        # Lista de sentenças
        lista_sentenca = ['P101 p102 p103 p104 p105 p106 p107 p108 p109 p110 .', 'p201 p202 p203 p204 p205 p206 p207 p208 p209 p210 .', 'p301 p302 p303 p304 p305 p306 p307 p308 p309 p310 .']

        # Divide a sentença em palavras pelos espaços.
        lista = []
        for i, sentenca in enumerate(lista_sentenca):
            # Divide a sentença pelos espaços em branco.
            lista.append(sentenca.split(' '))
        
        # Lista das janelas para comparar os elementos.
        lista_saida_janela = []
        lista_saida_conta_itens = []
        lista_saida_centro = []
        lista_saida_indice = []
        #for i, x in enumerate(lista.iterrows():
        for i, x in enumerate(lista):
            lista_janela, string_janela, lista_indice_janela, centro_janela = getJanelaLista(lista, tamanho_janela, i, trunca_janela)
            lista_saida_janela.append(lista_janela)
            lista_saida_conta_itens.append(contaItensLista(lista_janela))
            lista_saida_centro.append(centro_janela)
            lista_saida_indice.append(lista_indice_janela)
        
        lista_esperada_janela = [[['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', '.']], [['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', '.']], [['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', '.']]]
        lista_esperada_conta_itens = [33, 33, 33]
        lista_esperada_centro = [[1], [1], [1]]
        lista_esperada_indice = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
        self.assertListEqual(lista_saida_conta_itens, lista_esperada_conta_itens)
        self.assertListEqual(lista_saida_centro, lista_esperada_centro)
        self.assertListEqual(lista_saida_indice, lista_esperada_indice)   
        
    # Testes getJanelaSentenca3DocumentoMenorJanela
    def test_getJanelaSentenca3DocumentoMenorJanela(self):
        logger.info("Testando o getJanelaSentenca3 com lista de documento menor que janela ")
        
        # Valores de entrada
        tamanho_janela = 2
        trunca_janela = 30

        # Lista de sentenças
        lista_sentenca = ['P101 p102 p103 p104 p105 p106 p107 p108 p109 p110 .']

        # Divide a sentença em palavras pelos espaços.
        lista = []
        for i, sentenca in enumerate(lista_sentenca):
            # Divide a sentença pelos espaços em branco.
            lista.append(sentenca.split(' '))
        
        # Lista das janelas para comparar os elementos.
        lista_saida_janela = []
        lista_saida_conta_itens = []
        lista_saida_centro = []
        lista_saida_indice = []
        
        for i, x in enumerate(lista):
            lista_janela, string_janela, lista_indice_janela, centro_janela = getJanelaLista(lista, tamanho_janela, i, trunca_janela)
            lista_saida_janela.append(lista_janela)
            lista_saida_conta_itens.append(contaItensLista(lista_janela))
            lista_saida_centro.append(centro_janela)
            lista_saida_indice.append(lista_indice_janela)
        
        lista_esperada_janela = [[['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.']]]
        lista_esperada_conta_itens = [11]
        lista_esperada_centro = [[0]]
        lista_esperada_indice = [[0]]

        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
        self.assertListEqual(lista_saida_conta_itens, lista_esperada_conta_itens)
        self.assertListEqual(lista_saida_centro, lista_esperada_centro)
        self.assertListEqual(lista_saida_indice, lista_esperada_indice)
                
    # Testes getJanelaSentenca5DocumentoMaior
    def test_getJanelaSentenca5DocumentoMaior(self):
        logger.info("Testando o getJanelaSentenca5 com lista de documento maior que a janela ")
        
        # Valores de entrada
        tamanho_janela = 5
        trunca_janela = 30

        # Lista de sentenças
        lista_sentenca = ['P101 p102 p103 p104 p105 p106 p107 p108 p109 p110 p111 p112 p113 p114 p115 p116 p117 p118 p119 p120 .',
                        'p201 p202 p203 p204 p205 p206 p207 p208 p209 p210 p211 p212 p213 p214 p215 p216 p217 p218 p219 p220 p221 p222 p223 p224 .',
                        'p301 p302 p303 p304 p305 p306 p307 p308 p309 p310 p311 p312 p313 p314 p315 p316 p317 p318 p319 p320 p321 p322 p323 p324 p325 .',
                        'p401 p402 p403 p404 p405 p406 p407 p408 p409 p410 p411 p412 p413 p414 p415 p416 p417 p418 p419 p420 p421 p422 p423 p424 p425 p426 .',
                        'p501 p502 p503 p504 p505 p506 p507 p508 p509 p510 p511 p512 p513 p514 p515 p516 p517 p518 p519 p520 p521 p522 p523 p524 p525 .',
                        'p601 p602 p603 p604 p605 p606 p607 p608 p609 p610 p611 p612 p613 p614 p615 p616 p617 p618 p619 p620 p621 p622 p623 p624 .',
                        'p701 p702 p703 p704 p705 p706 p707 p708 p709 p710 p711 p712 p713 p714 p715 p716 p717 p718 p719 p720 p721 p722 p723 .']

        # Divide a sentença em palavras pelos espaços.
        lista = []
        for i, sentenca in enumerate(lista_sentenca):
            # Divide a sentença pelos espaços em branco.
            lista.append(sentenca.split(' '))
        
        # Lista das janelas para comparar os elementos.
        lista_saida_janela = []
        lista_saida_conta_itens = []
        lista_saida_centro = []
        lista_saida_indice = []
        for i, x in enumerate(lista):
            lista_janela, string_janela, lista_indice_janela, centro_janela = getJanelaLista(lista, tamanho_janela, i, trunca_janela)
            lista_saida_janela.append(lista_janela)
            lista_saida_conta_itens.append(contaItensLista(lista_janela))
            lista_saida_centro.append(centro_janela)
            lista_saida_indice.append(lista_indice_janela)
        
        lista_esperada_janela = [[['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', 'p111', 'p112', 'p113', 'p114', 'p115', 'p116', 'p117', 'p118', 'p119', 'p120', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209'], []], [['p120', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', 'p211', 'p212', 'p213', 'p214', 'p215', 'p216', 'p217', 'p218', 'p219', 'p220', 'p221', 'p222', 'p223', 'p224', '.'], ['p301', 'p302', 'p303'], []], [[], ['p224', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', 'p311', 'p312', 'p313', 'p314', 'p315', 'p316', 'p317', 'p318', 'p319', 'p320', 'p321', 'p322', 'p323', 'p324', 'p325', '.'], ['p401', 'p402'], []], [[], ['.'], ['p401', 'p402', 'p403', 'p404', 'p405', 'p406', 'p407', 'p408', 'p409', 'p410', 'p411', 'p412', 'p413', 'p414', 'p415', 'p416', 'p417', 'p418', 'p419', 'p420', 'p421', 'p422', 'p423', 'p424', 'p425', 'p426', '.'], ['p501', 'p502'], []], [[], ['p426', '.'], ['p501', 'p502', 'p503', 'p504', 'p505', 'p506', 'p507', 'p508', 'p509', 'p510', 'p511', 'p512', 'p513', 'p514', 'p515', 'p516', 'p517', 'p518', 'p519', 'p520', 'p521', 'p522', 'p523', 'p524', 'p525', '.'], ['p601', 'p602'], []], [[], ['p525', '.'], ['p601', 'p602', 'p603', 'p604', 'p605', 'p606', 'p607', 'p608', 'p609', 'p610', 'p611', 'p612', 'p613', 'p614', 'p615', 'p616', 'p617', 'p618', 'p619', 'p620', 'p621', 'p622', 'p623', 'p624', '.'], ['p701', 'p702', 'p703']], [[], ['p620', 'p621', 'p622', 'p623', 'p624', '.'], ['p701', 'p702', 'p703', 'p704', 'p705', 'p706', 'p707', 'p708', 'p709', 'p710', 'p711', 'p712', 'p713', 'p714', 'p715', 'p716', 'p717', 'p718', 'p719', 'p720', 'p721', 'p722', 'p723', '.']]]
        lista_esperada_conta_itens = [30, 30, 30, 30, 30, 30, 30]
        lista_esperada_centro = [[0], [1], [2], [2], [2], [2], [2]]
        lista_esperada_indice = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6]]

        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
        self.assertListEqual(lista_saida_conta_itens, lista_esperada_conta_itens)        
        self.assertListEqual(lista_saida_centro, lista_esperada_centro)
        self.assertListEqual(lista_saida_indice, lista_esperada_indice)
        
    # Testes getJanelaSentenca5DocumentoIgualJanela
    def test_getJanelaSentenca5DocumentoIgualJanela(self):
        logger.info("Testando o getJanelaSentenca5 com lista de documento igual a janela ")
        
        # Valores de entrada
        tamanho_janela = 5
        trunca_janela = 30

        # Lista de sentenças
        lista_sentenca = ['P101 p102 p103 p104 p105 p106 p107 p108 p109 p110 .', 'p201 p202 p203 p204 p205 p206 p207 p208 p209 p210 .', 'p301 p302 p303 p304 p305 p306 p307 p308 p309 p310 .']

        # Divide a sentença em palavras pelos espaços.
        lista = []
        for i, sentenca in enumerate(lista_sentenca):
            # Divide a sentença pelos espaços em branco.
            lista.append(sentenca.split(' '))
        
        # Lista das janelas para comparar os elementos.
        lista_saida_janela = []
        lista_saida_conta_itens = []
        lista_saida_centro = []
        lista_saida_indice = []
        #for i, x in enumerate(lista.iterrows():
        for i, x in enumerate(lista):
            lista_janela, string_janela, lista_indice_janela, centro_janela = getJanelaLista(lista, tamanho_janela, i, trunca_janela)
            lista_saida_janela.append(lista_janela)
            lista_saida_conta_itens.append(contaItensLista(lista_janela))
            lista_saida_centro.append(centro_janela)
            lista_saida_indice.append(lista_indice_janela)
        
        lista_esperada_janela = [[['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', '.']], [['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', '.']], [['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.'], ['p201', 'p202', 'p203', 'p204', 'p205', 'p206', 'p207', 'p208', 'p209', 'p210', '.'], ['p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p309', 'p310', '.']]]
        lista_esperada_conta_itens = [33, 33, 33]
        lista_esperada_centro = [[1], [1], [1]]
        lista_esperada_indice = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
        self.assertListEqual(lista_saida_conta_itens, lista_esperada_conta_itens)
        self.assertListEqual(lista_saida_centro, lista_esperada_centro)
        self.assertListEqual(lista_saida_indice, lista_esperada_indice)   
        
    # Testes getJanelaSentenca5DocumentoMenorJanela
    def test_getJanelaSentenca5DocumentoMenorJanela(self):
        logger.info("Testando o getJanelaSentenca5 com lista de documento menor que janela ")
        
        # Valores de entrada
        tamanho_janela = 5
        trunca_janela = 30

        # Lista de sentenças
        lista_sentenca = ['P101 p102 p103 p104 p105 p106 p107 p108 p109 p110 .']

        # Divide a sentença em palavras pelos espaços.
        lista = []
        for i, sentenca in enumerate(lista_sentenca):
            # Divide a sentença pelos espaços em branco.
            lista.append(sentenca.split(' '))
        
        # Lista das janelas para comparar os elementos.
        lista_saida_janela = []
        lista_saida_conta_itens = []
        lista_saida_centro = []
        lista_saida_indice = []
        
        for i, x in enumerate(lista):
            lista_janela, string_janela, lista_indice_janela, centro_janela = getJanelaLista(lista, tamanho_janela, i, trunca_janela)
            lista_saida_janela.append(lista_janela)
            lista_saida_conta_itens.append(contaItensLista(lista_janela))
            lista_saida_centro.append(centro_janela)
            lista_saida_indice.append(lista_indice_janela)
        
        lista_esperada_janela = [[['P101', 'p102', 'p103', 'p104', 'p105', 'p106', 'p107', 'p108', 'p109', 'p110', '.']]]
        lista_esperada_conta_itens = [11]
        lista_esperada_centro = [[0]]
        lista_esperada_indice = [[0]]

        self.assertListEqual(lista_saida_janela, lista_esperada_janela)
        self.assertListEqual(lista_saida_conta_itens, lista_esperada_conta_itens)
        self.assertListEqual(lista_saida_centro, lista_esperada_centro)
        self.assertListEqual(lista_saida_indice, lista_esperada_indice)        
                
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Texto")
    unittest.main()
    