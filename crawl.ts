type Entity = {
  entity: {
    name: string;
    image: string;
  }
}
type Resp = {
  data: Entity[];
}
const size = 50000;
const res = await fetch(`https://kg.diffbot.com/kg/v3/dql?type=query&token=d8f934b749507bc6a9939848e77b380c&query=type%3APerson%20has:image%20get:name,image,employments&size=${size}`)
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
