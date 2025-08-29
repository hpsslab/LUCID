from typing import Literal, Tuple, Set
from pydantic import BaseModel, Field, model_validator

ENTITIES : Literal = Literal[
        'PROBLEM',
        'APPROACH',
        'MATERIAL',
        'PROCESS',
        'METRIC',
        'ADVANTAGE'
]

RELATIONS : Literal = Literal[
        "USED_IN",
        "SOLVES",
        "EVALUATES",
        "OFFERS"
]

VALIDATION_SCHEMA : set(tuple([ENTITIES, RELATIONS, ENTITIES])) = {
        ("MATERIAL", "USED_IN", "APPROACH"),
        ("PROCESS", "USED_IN", "APPROACH"),
        ("APPROACH", "SOLVES", "PROBLEM"),
        ("APPROACH", "EVALUATES", "METRIC"),
        ("APPROACH", "OFFERS", "ADVANTAGE")
}

# Nodes must match one of the types in the ENTITIES variable and have a name.
class KGNode(BaseModel):
    label : ENTITIES
    name : str

    def __str__(self):
        return f"{label}: {name}"

# Relations must match one of the types seen in the RELATIONS variable and connect two KGNode objects.
class KGRelation(BaseModel):
    label : RELATIONS
    source_node : KGNode
    destination_node : KGNode

    def __str__(self):
        return f"{source_node} {label} {destination_node}"

# Lastly, the schema contains a list of nodes and relations and must match the validation schema.
class KGSchema(BaseModel):
    relations : list[KGRelation]
    
    # Use pydantic model validator to check triplets against validation schema
    @model_validator(mode = "after")
    def validate_relations(self):
        for triplet in self.relations:
            triplet = (triplet.source_node.label, triplet.label, triplet.destination_node.label)
            if triplet not in VALIDATION_SCHEMA:
                raise ValueError(f"Triplet {triplet} not allowed according to validation schema.")
        return self
    
    def __str__(self):
        return f"Relations: {relations}"

# TESTING ONLY
if __name__ == "__main__":
    nodes = [KGNode(label = "MATERIAL", name = "Gold"), KGNode(label = "APPROACH", name = "currency")]
    kg = KGSchema(
            relations = [KGRelation(label = "USED_IN", source_node = nodes[0], destination_node = nodes[1])]
    )
