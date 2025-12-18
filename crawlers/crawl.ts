type Entity = {
  entity: {
    name: string;
    image: string;
  }
}
type Resp = {
  data: Entity[];
}

const DIFFBOT_TOKEN = process.env.DIFFBOT_TOKEN;
if (!DIFFBOT_TOKEN) {
  throw new Error('DIFFBOT_TOKEN environment variable is required');
}

const size = 50000;
const res = await fetch(`https://kg.diffbot.com/kg/v3/dql?type=query&token=${DIFFBOT_TOKEN}&query=type%3APerson%20has:image%20get:name,image,employments&size=${size}`)
const data = await res.json() as Resp;

// Extract entities with job titles

const entities = data.data.map(entity => {
  const currentEmployment = entity.entity.employments?.[0];
  return {
    name: entity.entity.name,
    image: entity.entity.image,
    title: currentEmployment?.title || "Unknown"
  }
});

await Bun.write('entities_new.json', JSON.stringify(entities, null, 2));
