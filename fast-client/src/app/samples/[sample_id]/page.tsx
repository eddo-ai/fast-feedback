export default function SampleDetail({ params }: { params: { sample_id: string } }) {
  return <div>Sample detail page for {params.sample_id}</div>;
}
