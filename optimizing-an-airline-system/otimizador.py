# Prevendo a melhor combinação de pares em tripulação de aviões

# ***** Esta é a versão 2.0 deste script, atualizado em 02/07/2017 *****

import pandas as pd
import sys


# Dataset
df = pd.read_csv('flightdata.csv', header = None)

class Node:
	def __init__(self, num = None, child = None):
		self.num = num
		self.child = []
	def add_node(self, num):
		self.child.append(num)

# Solução recursiva para encontrar todos os conjuntos de pares
def find_child_recursive(lst_of_nodes):
	
	if lst_of_nodes[0][-1] == 175:
		
		possible_crews.append(lst_of_nodes.pop(0))
		if lst:
			find_child(lst)
		
	else:
		dup_len = len(access_flight[lst_of_nodes[0][-1]].child)-1
		temp = lst[0]
		for i in range(dup_len):
			temp = temp[:]
			lst.insert(0,temp)
		j = 0
		
		while j <= dup_len:
			
			lst[j].append(access_flight[lst_of_nodes[j][-1]].child[j])
			j += 1
		
		find_child(lst)

# Solução iterativa para encontrar todos os conjuntos de pares
def find_child_iterative(lst_of_nodes):
	while (not not lst_of_nodes):
		
		if len(lst_of_nodes[0])>6:
			lst_of_nodes.pop(0)
			continue
		if lst_of_nodes[0][-1] == 174:
			
			possible_crews.append(lst_of_nodes.pop(0))
		dup_len = 0
		try:
			
			dup_len = len(access_flight[lst_of_nodes[0][-1]].child)-1
			temp = lst_of_nodes[0]
		except:
			continue
		for i in range(dup_len):
			temp = temp[:]
			lst_of_nodes.insert(0,temp)
		j = 0
		
		while j <= dup_len:
			
			lst_of_nodes[j].append(access_flight[lst_of_nodes[j][-1]].child[j])
			j+=1

# Função representando o modelo com possível soluções viáveis
def find_crews(possible_crews, visited, crew):
	i=0
	while i < len(possible_crews):
		add = True
		for visit in visited:
			if visit in possible_crews[i]:
				add = False
				break
		if add:
			crew.append(possible_crews[i])
			for flight in possible_crews[i]:
				if flight not in [0,174]:
					visited.append(flight)
		i += 1

# Lista de nodes
a0=Node(0)
a1=Node(1)
a2=Node(2)
a3=Node(3)
a4=Node(4)
a5=Node(5)
a6=Node(6)
a7=Node(7)
a8=Node(8)
a9=Node(9)
a10=Node(10)
a11=Node(11)
a12=Node(12)
a13=Node(13)
a14=Node(14)
a15=Node(15)
a16=Node(16)
a17=Node(17)
a18=Node(18)
a19=Node(19)
a20=Node(20)
a21=Node(21)
a22=Node(22)
a23=Node(23)
a24=Node(24)
a25=Node(25)
a26=Node(26)
a27=Node(27)
a28=Node(28)
a29=Node(29)
a30=Node(30)
a31=Node(31)
a32=Node(32)
a33=Node(33)
a34=Node(34)
a35=Node(35)
a36=Node(36)
a37=Node(37)
a38=Node(38)
a39=Node(39)
a40=Node(40)
a41=Node(41)
a42=Node(42)
a43=Node(43)
a44=Node(44)
a45=Node(45)
a46=Node(46)
a47=Node(47)
a48=Node(48)
a49=Node(49)
a50=Node(50)
a51=Node(51)
a52=Node(52)
a53=Node(53)
a54=Node(54)
a55=Node(55)
a56=Node(56)
a57=Node(57)
a58=Node(58)
a59=Node(59)
a60=Node(60)
a61=Node(61)
a62=Node(62)
a63=Node(63)
a64=Node(64)
a65=Node(65)
a66=Node(66)
a67=Node(67)
a68=Node(68)
a69=Node(69)
a70=Node(70)
a71=Node(71)
a72=Node(72)
a73=Node(73)
a74=Node(74)
a75=Node(75)
a76=Node(76)
a77=Node(77)
a78=Node(78)
a79=Node(79)
a80=Node(80)
a81=Node(81)
a82=Node(82)
a83=Node(83)
a84=Node(84)
a85=Node(85)
a86=Node(86)
a87=Node(87)
a88=Node(88)
a89=Node(89)
a90=Node(90)
a91=Node(91)
a92=Node(92)
a93=Node(93)
a94=Node(94)
a95=Node(95)
a96=Node(96)
a97=Node(97)
a98=Node(98)
a99=Node(99)
a100=Node(100)
a101=Node(101)
a102=Node(102)
a103=Node(103)
a104=Node(104)
a105=Node(105)
a106=Node(106)
a107=Node(107)
a108=Node(108)
a109=Node(109)
a110=Node(110)
a111=Node(111)
a112=Node(112)
a113=Node(113)
a114=Node(114)
a115=Node(115)
a116=Node(116)
a117=Node(117)
a118=Node(118)
a119=Node(119)
a120=Node(120)
a121=Node(121)
a122=Node(122)
a123=Node(123)
a124=Node(124)
a125=Node(125)
a126=Node(126)
a127=Node(127)
a128=Node(128)
a129=Node(129)
a130=Node(130)
a131=Node(131)
a132=Node(132)
a133=Node(133)
a134=Node(134)
a135=Node(135)
a136=Node(136)
a137=Node(137)
a138=Node(138)
a139=Node(139)
a140=Node(140)
a141=Node(141)
a142=Node(142)
a143=Node(143)
a144=Node(144)
a145=Node(145)
a146=Node(146)
a147=Node(147)
a148=Node(148)
a149=Node(149)
a150=Node(150)
a151=Node(151)
a152=Node(152)
a153=Node(153)
a154=Node(154)
a155=Node(155)
a156=Node(156)
a157=Node(157)
a158=Node(158)
a159=Node(159)
a160=Node(160)
a161=Node(161)
a162=Node(162)
a163=Node(163)
a164=Node(164)
a165=Node(165)
a166=Node(166)
a167=Node(167)
a168=Node(168)
a169=Node(169)
a170=Node(170)
a171=Node(171)
a172=Node(172)
a173=Node(173) 
a174=Node(174)

flights = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,a64,a65,a66,a67,a68,a69,a70,a71,a72,a73,a74,a75,a76,a77,a78,a79,a80,a81,a82,a83,a84,a85,a86,a87,a88,a89,a90,a91,a92,a93,a94,a95,a96,a97,a98,a99,a100,a101,a102,a103,a104,a105,a106,a107,a108,a109,a110,a111,a112,a113,a114,a115,a116,a117,a118,a119,a120,a121,a122,a123,a124,a125,a126,a127,a128,a129,a130,a131,a132,a133,a134,a135,a136,a137,a138,a139,a140,a141,a142,a143,a144,a145,a146,a147,a148,a149,a150,a151,a152,a153,a154,a155,a156,a157,a158,a159,a160,a161,a162,a163,a164,a165,a166,a167,a168,a169,a170,a171,a172,a173,a174]
access_flight = {0:a0,1:a1,2:a2,3:a3,4:a4,5:a5,6:a6,7:a7,8:a8,9:a9,10:a10,11:a11,12:a12,13:a13,14:a14,15:a15,16:a16,17:a17,18:a18,19:a19,20:a20,21:a21,22:a22,23:a23,24:a24,25:a25,26:a26,27:a27,28:a28,29:a29,30:a30,31:a31,32:a32,33:a33,34:a34,35:a35,36:a36,37:a37,38:a38,39:a39,40:a40,41:a41,42:a42,43:a43,44:a44,45:a45,46:a46,47:a47,48:a48,49:a49,50:a50,51:a51,52:a52,53:a53,54:a54,55:a55,56:a56,57:a57,58:a58,59:a59,60:a60,61:a61,62:a62,63:a63,64:a64,65:a65,66:a66,67:a67,68:a68,69:a69,70:a70,71:a71,72:a72,73:a73,74:a74,75:a75,76:a76,77:a77,78:a78,79:a79,80:a80,81:a81,82:a82,83:a83,84:a84,85:a85,86:a86,87:a87,88:a88,89:a89,90:a90,91:a91,92:a92,93:a93,94:a94,95:a95,96:a96,97:a97,98:a98,99:a99,100:a100,101:a101,102:a102,103:a103,104:a104,105:a105,106:a106,107:a107,108:a108,109:a109,110:a110,111:a111,112:a112,113:a113,114:a114,115:a115,116:a116,117:a117,118:a118,119:a119,120:a120,121:a121,122:a122,123:a123,124:a124,125:a125,126:a126,127:a127,128:a128,129:a129,130:a130,131:a131,132:a132,133:a133,134:a134,135:a135,136:a136,137:a137,138:a138,139:a139,140:a140,141:a141,142:a142,143:a143,144:a144,145:a145,146:a146,147:a147,148:a148,149:a149,150:a150,151:a151,152:a152,153:a153,154:a154,155:a155,156:a156,157:a157,158:a158,159:a159,160:a160,161:a161,162:a162,163:a163,164:a164,165:a165,166:a166,167:a167,168:a168,169:a169,170:a170,171:a171,172:a172,173:a173,174:a174}

print ("Procesando o dataset")
for node in flights: 
	flight_num = node.num
	for j, val in df.iterrows():
		idx = -1
		for feasible in val:
			idx += 1
			if feasible==1 and flight_num==j:
				node.add_node(int(idx))
	print (flight_num)
	print (node.child)

# Toda tripulação possível	
possible_crews = [] 
lst = [[0]] 
find_child_iterative(lst)
solution = {}
sol_num = 1
possible_crews = sorted(possible_crews, key = len, reverse = True)
visited = [-1]
crew = []
i = len(possible_crews)
print ("Iniciando a busca das combinações de tripulação")
# Testando diferentes combinações - esse processo é demorado 
while i >= 1:
	if sol_num > 1:
		break
	i -= 1
	visited = [-1]
	crew = []
	temp = possible_crews[i:]+possible_crews[:i]
	find_crews(temp, visited, crew)
	if len(visited) > 173: 
		print (visited)
		crew.sort()
		add = True
		for key in solution:
			add=True
			if crew == solution[key]:
				add = False
				break
		if add:
			solution[sol_num] = crew
			sol_num += 1

print (solution)






