class Instruction:
    def __init__(self, inst_type, text, line_number, line_column, global_idx, description, relativ_pos):
        self.type = inst_type
        self.text = text
        self.line_number = line_number
        self.line_column = line_column
        self.global_idx = global_idx 
        self.description = description
        self.relativ_pos = relativ_pos

    def GetDescription(self):
        return self.description
        
class LinterReport:
    def __init__(self, rule_id, message, evidence, col_begin, col_end, line_begin, line_end, severity):
        self.rule_id = rule_id
        self.message = message
        self.evidence = evidence
        self.col_begin = col_begin
        self.col_end = col_end 
        self.line_begin = line_begin
        self.line_end = line_end
        self.severity = severity

class DataPoint: 
    def __init__(self, source_code, target_code, warning_line, linter_report, instructions, source_file, target_file, repo, source_filename, target_filename, source_changeid, target_changeid):

        self.source_code = source_code # string
        self.target_code = target_code # string
        self.warning_line = warning_line # string
        self.linter_report = linter_report # class LinterReport
        self.instructions = instructions # list of class Instruction
        self.source_file = source_file # string
        self.target_file = target_file # string
        self.repo = repo
        self.source_filename = source_filename # string
        self.target_filename = target_filename # string
        self.source_changeid = source_changeid # string
        self.target_changeid = target_changeid # string

    def GetDescription(self):
        desc = "WARNING\n" + self.linter_report.rule_id + " " + self.linter_report.message + " at line: " + str(self.linter_report.line_begin) + "\n"

        desc += "WARNING LINE\n" + self.warning_line + "\n"
        desc += "SOURCE PATCH\n" + self.source_code + "\nTARGET PATCH\n" + self.target_code + "\n"
        
        desc += "INSTRUCTIONS\n"
        for inst in self.instructions:
            desc += inst.GetDescription() +"\n"
        return desc
    
    def GetT5Representation(self, include_warning):
        if include_warning:
            inputs = 'fix ' + self.linter_report.rule_id + ' ' + self.linter_report.message + ' ' + self.warning_line + ':\n' + self.source_code + ' </s>'
        else:
            inputs = 'fix ' + self.source_code + ' </s>'
        outputs = self.target_code + ' </s>'
        return inputs, outputs 

        

import json
def GetDataAsPython(data_json_path):
    with open(data_json_path, 'r', errors='ignore') as f:
        data_json = json.load(f)

    # converts a data point in json format to a data point in python object
    def FromJsonToPython(sample):
        linter_report = LinterReport(sample["linter_report"]["rule_id"], sample["linter_report"]["message"], sample["linter_report"]["evidence"], sample["linter_report"]["col_begin"], sample["linter_report"]["col_end"], sample["linter_report"]["line_begin"], sample["linter_report"]["line_end"], sample["linter_report"]["severity"])

        instructions = []
        for inst in sample["instructions"]:
            instruction = Instruction(inst["type"], inst["text"], inst["line_number"], inst["line_column"], inst["global_idx"], inst["description"], inst["relativ_pos"])
            instructions.append(instruction)
        
        
        data_point = DataPoint(sample["source_code"], sample["target_code"], sample["warning_line"], linter_report, instructions, sample["source_file"], sample["target_file"], sample["repo"], sample["source_filename"], sample["target_filename"], sample["source_changeid"], sample["target_changeid"])
        
        return data_point

    data = [FromJsonToPython(sample) for sample in data_json]
    return data
