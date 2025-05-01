import json
from py2neo import Graph
from tqdm import tqdm


def read_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        data = json.loads(line)
        print(type(data))
        print(json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False))
        print(type(json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)))
        break


class MedicalExtractor(object):
    def __init__(self):
        super().__init__()
        self.graph = Graph("http://localhost:7474/", auth=("neo4j", "1234"))
        # 第一次要清空数据库
        # self.graph.delete_all()
        # 一共4类节点
        self.drugs = []
        self.foods = []
        self.diseases = []
        self.symptoms = []  # 症状

        # 构建节点实体信息
        self.rels_noeat = []
        self.rels_doeat = []
        self.rels_recommenddrug = []
        self.rels_symptoms = []  # 疾病症状关系

    def extract_triple(self, data_path):
        with open(data_path, "r", encoding="utf-8") as fr:
            for line in tqdm(fr.readlines()):
                data_json = json.loads(line)
                disease = data_json["name"]
                self.diseases.append(disease)

                # 判断症状是否存在字典里面, 如果存在, 需要获取所有的症状, 并且定义所有的疾病-症状关系
                if "symptom" in data_json:
                    self.symptoms += data_json["symptom"]
                    for symp in data_json["symptom"]:
                        self.rels_symptoms.append([disease, "has_symptom", symp])

                # 判断并发症是否存在
                if "acompany" in data_json:
                    for acom in data_json["acompany"]:
                        self.diseases.append(acom)

                # 判断推荐药物是否存在
                if "recommand_drug" in data_json:
                    self.drugs += data_json["recommand_drug"]
                    for drug in data_json["recommand_drug"]:
                        self.rels_recommenddrug.append([disease, "recommand_drug", drug])

                # 判断事务是否存在
                if "not_eat" in data_json:
                    self.foods += data_json["not_eat"]
                    for _not in data_json["not_eat"]:
                        self.rels_noeat.append([disease, "not_eat", _not])

                    self.foods += data_json["do_eat"]
                    for _do in data_json["do_eat"]:
                        self.rels_doeat.append([disease, "do_eat", _do])

                # 判断药物
                if "drug_detail" in data_json:
                    for det in data_json["drug_detail"]:
                        det_split = det.split("(")
                        if len(det_split) == 2:
                            p, d = det_split
                            d = d.rstrip(")")
                            self.drugs.append(d)
                        else:
                            d = det_split[0]
                            self.drugs.append(d)

    def write_nodes(self, entities, entity_type):
        print(f"写入{entity_type}实体")
        for node in tqdm(set(entities)):
            # cql = """MERGE (n:{label}{{name:'{entity_name}'}})""".format(label=entity_type, entity_name=node.replace("'", ""))
            cql = f"""MERGE (n:{entity_type}{{name:'{node.replace("'", "")}'}})"""
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def create_entities(self):
        self.write_nodes(self.drugs, "药品")
        self.write_nodes(self.symptoms, "症状")
        self.write_nodes(self.foods, "食物")
        self.write_nodes(self.diseases, "疾病")

    def write_edges(self, triple, head_type, tail_type):
        print(f"写入{triple[0][1]}关系")
        for head, relation, tail in tqdm(triple):
            cql = f"MATCH (p:{head_type}), (q:{tail_type}) WHERE p.name='{head}' AND q.name='{tail}' MERGE (p)-[r:{relation}]->(q)"
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def create_relations(self):
        self.write_edges(self.rels_noeat, "疾病", "食物")
        self.write_edges(self.rels_doeat, "疾病", "食物")
        self.write_edges(self.rels_symptoms, "疾病", "症状")
        self.write_edges(self.rels_recommenddrug, "疾病", "药品")


if __name__ == '__main__':
    # read_data("./medical.json")
    me = MedicalExtractor()
    me.extract_triple("./medical.json")
    me.create_entities()
    me.create_relations()

