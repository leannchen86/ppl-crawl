type Entity = {
  entity: {
    name: string;
    image: string;
  }
}
type Resp = {
  data: Entity[];
}
const size = 30000;
const res = await fetch(`https://kg.diffbot.com/kg/v3/dql?type=query&token=d8f934b749507bc6a9939848e77b380c&query=type%3APerson%20has:image%20get:name,image&size=${size}`)
const data = await res.json() as Resp;

const entities = data.data.map(entity => {
  return {
    name: entity.entity.name,
    image: entity.entity.image,
  }
});

await Bun.write('entities.json', JSON.stringify(entities, null, 2));
